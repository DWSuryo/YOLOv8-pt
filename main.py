import copy
import csv
import os
import warnings
import argparse
from datetime import datetime
import zipfile

import torch
import tqdm
import yaml
from torch.utils import data
import cv2
import numpy as np
import time
import statistics

from nets import nn
from utils import util_mod as util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

from plotting import plot_mAP

data_dir = 'D:/dataset/coco-2017-download'

def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    # Model
    # model = nn.yolo_v8_n(len(params['names'].values())).cuda()
    version = args.version
    if version == 'n':
        model = nn.yolo_v8_n(len(params['names']))
    elif version == 's':
        model = nn.yolo_v8_s(len(params['names']))
    elif version == 'm':
        model = nn.yolo_v8_m(len(params['names']))
    elif version == 'l':
        model = nn.yolo_v8_l(len(params['names']))
    elif version == 'x':
        model = nn.yolo_v8_x(len(params['names']))
    else:
        raise ValueError(f"Unsupported YOLOv8 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")
    # model = nn.yolo_v8_m(len(params['names']))
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open(f'{data_dir}/train2017.txt') as reader:
        for filename in reader.readlines():
            # filename = filename.rstrip().split('/')[-1]
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/train2017/' + filename)
        print("filename lists: ", len(filenames))
    # check if file exists
    existing_count = 0
    nonexisting_count = 0

    for filepath in filenames:
        if os.path.exists(filepath):
            existing_count += 1
        else:
            nonexisting_count += 1

    print(f"Number of existing files: {existing_count}")
    print(f"Number of non-existing files: {nonexisting_count}")


    # if args.world_size <= 1:
    #     sampler = None
    # else:
    #     sampler = data.distributed.DistributedSampler(dataset)
    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    num_batch = len(loader)
    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start training
    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)

    # --- MODIFICATION: Use args.save_dir ---
    csv_file = os.path.join(args.save_dir, 'results.csv')

    # Initialize lists to record mAP and epoch numbers
    mAP_list = []
    epoch_list = []
    # Write file
    with open(csv_file, 'w', newline='') as log:
        if args.local_rank == 0:
            writer = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl', 'train_loss',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_batch)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            # m_loss = util.AverageMeter()

            for i, (samples, targets) in p_bar:
                # print(targets.keys())
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255
                # targets = targets.cpu()

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, np.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = np.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = np.interp(x, xp, fp)

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    # loss = criterion(outputs, targets)
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                # m_loss.update(loss.item(), samples.size(0))

                # loss *= args.batch_size  # loss scaled by batch_size
                # loss *= args.world_size  # gradient averaged between devices in DDP mode
                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

                # del loss
                # del outputs

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                current_mAP = last[0]  # mAP computed from test()
                mAP_list.append(current_mAP)
                epoch_list.append(epoch + 1)
                total_train_loss = avg_box_loss.avg + avg_cls_loss.avg + avg_dfl_loss.avg

                writer.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'train_loss': str(f'{total_train_loss:.3f}'),
                                 'mAP': str(f'{last[0]:.6f}'),
                                 'mAP@50': str(f'{last[1]:.6f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                if current_mAP > best:
                    best = current_mAP

                # Save model
                ckpt = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema).half()
                        }

                # Save last, best and delete
                # torch.save(ckpt, './weights/last.pt')
                # if best == last[1]:
                #     torch.save(ckpt, './weights/best.pt')
                # del ckpt

                # --- MODIFICATION: Save to specific dir with clean names ---
                last_path = os.path.join(args.save_dir, 'last.pt')
                best_path = os.path.join(args.save_dir, 'best.pt')

                torch.save(ckpt, f=last_path)
                # if best == last[0]:
                if best == current_mAP:
                    torch.save(ckpt, f=best_path)
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer(f'./weights/{version}{args.epochs}/last.pt', args)  # strip optimizers
        util.strip_optimizer(f'./weights/{version}{args.epochs}/best.pt', args)  # strip optimizers

    torch.cuda.empty_cache()
    plot_mAP(args)


@torch.no_grad()
def test(args, params, model=None):
    version = args.version
    epochs = args.epochs
    filenames = []
    with open(f'{data_dir}/val2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(f'{data_dir}/images/val2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if model is None:
        plot = True
        # model = torch.load(f'./weights/best_{version}_{epochs}.pt', map_location='cuda', weights_only=False)['model'].float()
        
        # --- MODIFICATION: Load from save_dir ---
        path = os.path.join(args.save_dir, "best.pt")
        print(f"Testing model: {path}")
        model = torch.load(f=path, map_location='cuda', weights_only=False)
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        # targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = model(samples)

        # NMS
        scale = torch.tensor((width, height, width, height)).cuda()  # to pixels
        outputs = util.non_max_suppression(outputs, 0.001, 0.65)

        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

        # for i, output in enumerate(outputs):
        #     labels = targets[targets[:, 0] == i, 1:]
        #     correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

        #     if output.shape[0] == 0:
        #         if labels.shape[0]:
        #             metrics.append((correct, *torch.zeros((3, 0)).cuda()))
        #         continue

        #     detections = output.clone()
        #     util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

        #     # Evaluate
        #     if labels.shape[0]:
        #         tbox = labels[:, 1:5].clone()  # target boxes
        #         tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
        #         tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
        #         tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
        #         tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
        #         util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

        #         correct = np.zeros((detections.shape[0], iou_v.shape[0]))
        #         correct = correct.astype(bool)

        #         t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
        #         iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
        #         correct_class = t_tensor[:, 0:1] == detections[:, 5]
        #         for j in range(len(iou_v)):
        #             x = torch.where((iou >= iou_v[j]) & correct_class)
        #             if x[0].shape[0]:
        #                 matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
        #                 matches = matches.cpu().np()
        #                 if x[0].shape[0] > 1:
        #                     matches = matches[matches[:, 2].argsort()[::-1]]
        #                     matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        #                     matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        #                 correct[matches[:, 1].astype(int), j] = True
        #         correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
        #     metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to np
    if len(metrics) and metrics[0].any():
        # tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
        # --- MODIFICATION: Pass save_dir to compute_ap ---
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(
            version,
            epochs, 
            *metrics, 
            plot=plot, 
            names=params["names"],
            save_dir=args.save_dir  # <--- PASS THIS HERE
        )

    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))

    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre

def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    print(f"params amount: {len(params['names'])}")
    version = args.version
    if version == 'n':
        model = nn.yolo_v8_n(len(params['names'])).fuse()
    elif version == 's':
        model = nn.yolo_v8_s(len(params['names'])).fuse()
    elif version == 'm':
        model = nn.yolo_v8_m(len(params['names'])).fuse()
    elif version == 'l':
        model = nn.yolo_v8_l(len(params['names'])).fuse()
    elif version == 'x':
        model = nn.yolo_v8_x(len(params['names'])).fuse()
    else:
        raise ValueError(f"Unsupported YOLOv8 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")
    # model = nn.yolo_v8_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')

def zip_weights_directory(args):
    # Zip the specific folder (e.g., weights/n5)
    target_dir = args.save_dir
    output_zip = f"{args.save_dir}.zip" # e.g., weights/n5.zip

    if not os.path.exists(target_dir):
        print(f"Error: {target_dir} does not exist.")
        return

    import shutil
    shutil.make_archive(args.save_dir, 'zip', target_dir)
    print(f"Successfully zipped {target_dir} to {output_zip}")

def inference(model, args, params):
    source_type = args.inference
    if source_type == "image":
        source_path = f"./src/stadium_crowd.jpg"
        frame = cv2.imread(source_path)

        if frame is None:
            print(f"Error: Could not read image from {source_path}")
            return
        
        # Start timing for single image inference
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Preprocessing, Inference, and Post-processing for a single image
        # 1. Pre-process (Don't time this if checking Model Speed)
        image = frame.copy()
        shape = image.shape[:2]

        r = args.input_size / max(shape[0], shape[1])
        if r != 1:
            resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
        height, width = image.shape[:2]

        # Scale ratio (new / old)
        r = min(1.0, args.input_size / height, args.input_size / width)

        # Compute padding
        pad = int(round(width * r)), int(round(height * r))
        w = (args.input_size - pad[0]) / 2
        h = (args.input_size - pad[1]) / 2

        if (width, height) != pad:
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

        # Convert HWC to CHW, BGR to RGB
        x = image.transpose((2, 0, 1))[::-1]
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x).unsqueeze(dim=0).cuda().half()/255.0
        # x = x.unsqueeze(dim=0)
        # x = x.cuda()
        # x = x.half()
        # x = x / 255

        # 2. Pure Inference Timer
        start_event.record()
        outputs = model(x)
        end_event.record()
        torch.cuda.synchronize()
        model_latency = start_event.elapsed_time(end_event)

        # 3. Post-process (NMS) Timer
        t0 = time.time()
        outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
        nms_time = (time.time() - t0) * 1000
        
        # End timing and calculate latency
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)

        # Total System Latency
        total_latency = model_latency + nms_time
        
        if outputs is not None:
            outputs[:, [0, 2]] -= w
            outputs[:, [1, 3]] -= h
            outputs[:, :4] /= min(height / shape[0], width / shape[1])
            outputs[:, 0].clamp_(0, shape[1])
            outputs[:, 1].clamp_(0, shape[0])
            outputs[:, 2].clamp_(0, shape[1])
            outputs[:, 3].clamp_(0, shape[0])
            for box in outputs:
                box = box.cpu().numpy()
                x1, y1, x2, y2, score, index = box
                class_name = params['names'][int(index)]
                label = f"{class_name} {score:.2f}"
                util.draw_box(frame, box, index, label)

        # Display latency on the image
        latency_text = f"Model: {model_latency:.1f}ms | NMS: {nms_time:.1f}ms | Total: {total_latency:.1f}ms"
        cv2.putText(frame, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Inference Result', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # The existing code for video and camera inference (which works)
        # This part remains unchanged
        model = torch.load(f'./weights/{args.version}{args.epochs}/best.pt', 'cuda', weights_only=False)['model'].float()
        model.half()
        model.eval()

        if source_type == "video":
            camera = cv2.VideoCapture('src/crowd1.mp4')
        elif source_type == "camera":
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Get video properties
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = camera.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output2.mp4', fourcc, fps, (width, height))

        if not camera.isOpened():
            print("Error opening video stream or file")
            return

        start_time = datetime.now()
        frame_count = 0
        fps_display = 0.0
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        while camera.isOpened():
            success, frame = camera.read()
            if success:
                frame_count += 1
                current_time = datetime.now()
                elapsed_time = (current_time - start_time).total_seconds()
                if elapsed_time > 1.0:
                    fps_display = frame_count / elapsed_time
                    frame_count = 0
                    start_time = current_time

                start_event.record()

                # --- TIMER 1: Start System Timer ---
                t_start_system = time.time()

                # 1. Pre-processing (CPU)
                t_prep_start = time.time()
                image = frame.copy()

                shape = image.shape[:2]
                r = args.input_size / max(shape[0], shape[1])
                if r != 1:
                    resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                    image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
                height, width = image.shape[:2]
                r = min(1.0, args.input_size / height, args.input_size / width)
                pad = int(round(width * r)), int(round(height * r))
                w = (args.input_size - pad[0]) / 2
                h = (args.input_size - pad[1]) / 2
                if (width, height) != pad:
                    image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
                top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
                left, right = int(round(w - 0.1)), int(round(w + 0.1))
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
                x = image.transpose((2, 0, 1))[::-1]
                x = np.ascontiguousarray(x)

                x = torch.from_numpy(x).unsqueeze(dim=0).cuda().half() / 255
                # x = x.unsqueeze(dim=0)
                # x = x.cuda()
                # x = x.half()
                # x = x / 255
                t_prep_end = time.time()

                # 2. Inference (GPU)
                # We use CUDA events for precise GPU timing
                start_event = torch.Event('cuda', enable_timing=True)
                end_event = torch.Event('cuda', enable_timing=True)
                
                start_event.record()

                outputs = model(x)
                end_event.record()
                torch.cuda.synchronize() # Wait for GPU to finish
                inference_time_ms = start_event.elapsed_time(end_event) # Pure Model Time
                
                # 3. NMS (CPU)
                t_nms_start = time.time()
                outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
                t_nms_end = time.time()
                # Calculate Latencies
                preprocess_ms = (t_prep_end - t_prep_start) * 1000
                nms_ms = (t_nms_end - t_nms_start) * 1000
                e2e_latency_ms = preprocess_ms + inference_time_ms + nms_ms

                # 4. Visualization (CPU - Slow!)
                if outputs is not None:
                    outputs[:, [0, 2]] -= w
                    outputs[:, [1, 3]] -= h
                    outputs[:, :4] /= min(height / shape[0], width / shape[1])
                    outputs[:, 0].clamp_(0, shape[1])
                    outputs[:, 1].clamp_(0, shape[0])
                    outputs[:, 2].clamp_(0, shape[1])
                    outputs[:, 3].clamp_(0, shape[0])
                    for box in outputs:
                        box = box.cpu().numpy()
                        x1, y1, x2, y2, score, index = box
                        class_name = params['names'][int(index)]
                        label = f"{class_name} {score:.2f}"
                        util.draw_box(frame, box, index, label)
                
                # fps_text = f"FPS: {fps_display:.2f}"
                # latency_text = f"Latency: {latency_ms:.2f} ms"
                # cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # cv2.imshow('Frame', frame)
                # out.write(frame)

                # 5. FPS Calculation (Moving Average)
                # We use t_start_system to capture the FULL loop time
                system_latency_ms = (time.time() - t_start_system) * 1000
                current_fps = 1000.0 / (system_latency_ms + 1e-8)
                
                # Smoothing the FPS display so it doesn't flicker
                fps_display = 0.9 * fps_display + 0.1 * current_fps

                # --- DISPLAY STATS ---
                # Line 1: Real System Speed
                theoretical_fps = 1000.0 / e2e_latency_ms
                cv2.putText(frame, f"System FPS: {fps_display:.1f} | Model Potential: {theoretical_fps:.0f} FPS", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Line 2: The breakdown (Why is it slow?)
                info_text = f"Pre:{preprocess_ms:.1f}ms | Inf:{inference_time_ms:.1f}ms | NMS:{nms_ms:.1f}ms"
                cv2.putText(frame, info_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # # Line 3: Theoretical Max FPS (If you removed display/webcam bottleneck)
                # cv2.putText(frame, f"Model Potential: {theoretical_fps:.0f} FPS", (10, 90), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow('Inference', frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        camera.release()
        out.release()
        cv2.destroyAllWindows()

def benchmark(model, args, params):
    # Setup Input (Camera or Video)
    source_type = args.inference
    source_path = 'src/crowd1.mp4' # Default video
    
    if source_type == "camera":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print(f"📷 Source: Camera (0)")
    else:
        # Check if file exists
        if not os.path.exists(source_path):
            print(f"Error: Video file not found at {source_path}")
            return
        cap = cv2.VideoCapture(source_path)
        print(f"🎬 Source: Video ({source_path})")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0: total_frames = "Unknown" # Camera doesn't have total frames

    print(f"\n{'='*60}")
    print(f"🚀 BENCHMARK CONFIGURATION")
    print(f"   Model: {args.version.upper()} @ {args.input_size}px")
    print(f"   Display: {'ON' if args.view else 'OFF'}")
    print(f"   Time Limit: {args.timeout if args.timeout else 'None'} seconds")
    print(f"{'='*60}\n")

    records = {'pre': [], 'inf': [], 'nms': [], 'total': []}
    
    # Timers
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    benchmark_start_time = time.time()
    
    frame_idx = 0
    warmup_frames = 30
    
    while cap.isOpened():
        # --- 1. Check Time Limit ---
        if args.timeout:
            elapsed_total = time.time() - benchmark_start_time
            if elapsed_total > args.timeout:
                print(f"\n⏰ Time limit of {args.timeout}s reached. Stopping.")
                break

        ret, frame = cap.read()
        if not ret:
            print("\nEnd of stream reached.")
            break
        
        frame_idx += 1

        # --- 2. Pre-process ---
        t0 = time.time()
        image = frame.copy()
        shape = image.shape[:2]
        
        # Letterbox Resize
        r = args.input_size / max(shape[0], shape[1])
        if r != 1:
            resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
        
        height, width = image.shape[:2]
        r = min(1.0, args.input_size / height, args.input_size / width)
        pad = int(round(width * r)), int(round(height * r))
        w = (args.input_size - pad[0]) / 2
        h = (args.input_size - pad[1]) / 2
        
        if (width, height) != pad:
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        
        x = image.transpose((2, 0, 1))[::-1]
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x).unsqueeze(0).cuda()
        
        # Adaptive Precision
        model_dtype = next(model.parameters()).dtype
        x = x.to(model_dtype)
        x = x / 255.0
        
        t1 = time.time()

        # --- 3. Inference ---
        start_event.record()
        outputs = model(x)
        end_event.record()
        torch.cuda.synchronize()
        
        # --- 4. NMS ---
        t2 = time.time()
        outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
        t3 = time.time()

        # --- 5. Data Recording ---
        if frame_idx > warmup_frames:
            t_pre = (t1 - t0) * 1000
            t_inf = start_event.elapsed_time(end_event)
            t_nms = (t3 - t2) * 1000
            t_total = t_pre + t_inf + t_nms

            records['pre'].append(t_pre)
            records['inf'].append(t_inf)
            records['nms'].append(t_nms)
            records['total'].append(t_total)

            if frame_idx % 50 == 0:
                 print(f"Frame {frame_idx} | Latency: {t_total:.2f}ms")

        # --- 6. Optional Visualization ---
        if args.view:
            if outputs is not None:
                # Rescale boxes back to original image size
                outputs[:, [0, 2]] -= w
                outputs[:, [1, 3]] -= h
                outputs[:, :4] /= min(height / shape[0], width / shape[1])
                outputs[:, 0].clamp_(0, shape[1])
                outputs[:, 1].clamp_(0, shape[0])
                outputs[:, 2].clamp_(0, shape[1])
                outputs[:, 3].clamp_(0, shape[0])
                
                for box in outputs:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    class_name = params['names'][int(index)]
                    label = f"{class_name} {score:.2f}"
                    util.draw_box(frame, box, index, label)
            
            # Show "BENCHMARKING" on screen
            cv2.putText(frame, "BENCHMARK MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Benchmark', frame)
            
            # Allow quitting with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted benchmark.")
                break

    cap.release()
    if args.view:
        cv2.destroyAllWindows()

    # --- Calculations & Saving (Same as before) ---
    count = len(records['total'])
    if count == 0:
        print("Not enough frames processed.")
        return

    stats = {}
    for key in records:
        data = records[key]
        stats[key] = {
            'min': min(data),
            'max': max(data),
            'avg': statistics.mean(data),
            'p95': statistics.quantiles(data, n=20)[-1]
        }
    
    avg_fps = 1000.0 / stats['total']['avg']

    print(f"\n{'='*60}")
    print(f"📊 RESULTS (Frames: {count} | View: {args.view})")
    print(f"{'='*60}")
    print(f"{'Metric':<15} | {'Avg (ms)':<10} | {'P95 (ms)':<10}")
    print(f"{'-'*60}")
    print(f"{'Pre-Process':<15} | {stats['pre']['avg']:<10.2f} | {stats['pre']['p95']:<10.2f}")
    print(f"{'Inference':<15} | {stats['inf']['avg']:<10.2f} | {stats['inf']['p95']:<10.2f}")
    print(f"{'NMS':<15} | {stats['nms']['avg']:<10.2f} | {stats['nms']['p95']:<10.2f}")
    print(f"{'-'*60}")
    print(f"Total Latency   | {stats['total']['avg']:<10.2f} | {stats['total']['p95']:<10.2f}")
    print(f"Potential FPS   | {avg_fps:.2f}")
    print(f"{'='*60}\n")

    # --- Save to CSV ---
    csv_filename = os.path.join(args.save_dir, f"benchmark_stats_{args.inference}.csv")
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Input Size", "Metric", "Avg(ms)", "Min(ms)", "Max(ms)", "P95(ms)", "FPS"])
        
        # Write rows
        base_info = [args.version, args.input_size]
        writer.writerow(base_info + ["Pre-Process", f"{stats['pre']['avg']:.2f}", f"{stats['pre']['min']:.2f}", f"{stats['pre']['max']:.2f}", f"{stats['pre']['p95']:.2f}", "-"])
        writer.writerow(base_info + ["Inference", f"{stats['inf']['avg']:.2f}", f"{stats['inf']['min']:.2f}", f"{stats['inf']['max']:.2f}", f"{stats['inf']['p95']:.2f}", "-"])
        writer.writerow(base_info + ["NMS", f"{stats['nms']['avg']:.2f}", f"{stats['nms']['min']:.2f}", f"{stats['nms']['max']:.2f}", f"{stats['nms']['p95']:.2f}", "-"])
        writer.writerow(base_info + ["Total E2E", f"{stats['total']['avg']:.2f}", f"{stats['total']['min']:.2f}", f"{stats['total']['max']:.2f}", f"{stats['total']['p95']:.2f}", f"{avg_fps:.2f}"])

    print(f"✅ Statistics saved to: {csv_filename}")


def main():
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--version', default='m', type=str)
    parser.add_argument('--zip', action='store_true')
    parser.add_argument("--inference", type=str, choices=["image", "video", "camera"])

    parser.add_argument('--benchmark', action='store_true', help="Run speed benchmark")
    # --- ADD THESE TWO ---
    parser.add_argument('--view', action='store_true', help="Show video during benchmark (slower)")
    parser.add_argument('--timeout', type=int, default=None, help="Stop benchmark after X seconds")
    parser.add_argument('--state-dict', action='store_true', help="Converts model into state_dict model")

    args = parser.parse_args()
    print(args)

    # --- STRATEGY: Define Dynamic Save Directory ---
    # Example result: ./weights/n5 or ./weights/s100
    run_name = f"{args.version}{args.epochs}"
    args.save_dir = os.path.join("weights", run_name)
    print(f"Output Directory: {args.save_dir}")
    # -----------------------------------------------

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()

    if args.zip:
        zip_weights_directory(args)

    if args.inference:
        # print(args.inference)
        # version = args.version
        # if version == 'n':
        #     model = nn.yolo_v8_n(len(params['names']))
        # elif version == 's':
        #     model = nn.yolo_v8_s(len(params['names']))
        # elif version == 'm':
        #     model = nn.yolo_v8_m(len(params['names']))
        # elif version == 'l':
        #     model = nn.yolo_v8_l(len(params['names']))
        # elif version == 'x':
        #     model = nn.yolo_v8_x(len(params['names']))
        # else:
        #     raise ValueError(f"Unsupported YOLOv8 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")
        # model_path = f"./weights/original/best.pt"
        model_path = os.path.join(args.save_dir, "best.pt")
        if not os.path.exists(model_path):
             print(f"Error: Model not found at {model_path}")
             return
        
        print(f"Loading model from: {model_path}")
        model_data = torch.load(model_path, map_location="cuda", weights_only=False)
        model = model_data["model"].eval().cuda().half()

        if args.benchmark:
            benchmark(model, args, params)
        else:
            inference(model, args, params)

    if args.state_dict:
        util.strip_optimizer(f'./weights/{args.version}{args.epochs}/last.pt', args)  # strip optimizers
        util.strip_optimizer(f'./weights/{args.version}{args.epochs}/best.pt', args)  # strip optimizers

    time_end = datetime.now()
    print("Finished at Date and Time:", time_end.strftime("%Y-%m-%d %H:%M:%S"))
    time_duration = time_end - time_start
    # Format the duration as Days HH:MM:SS
    days = time_duration.days
    seconds = time_duration.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_duration = f"{days} Days {hours:02}:{minutes:02}:{seconds:02}"
    print(f"Code execution time: {formatted_duration}")

if __name__ == "__main__":
    main()
