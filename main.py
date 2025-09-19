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

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

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
    # model = nn.yolo_v11_m(len(params['names']))
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
    # with open(f'weights/step.csv', 'w') as f:
    with open(f'weights/step_{version}_{args.epochs}.csv', 'w', newline='') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            # if args.world_size > 1:
            #     sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            # print(p_bar)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()
            m_loss = util.AverageMeter()

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
                    loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= args.batch_size  # loss scaled by batch_size
                loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()

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
                    s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg)
                    p_bar.set_description(s)

                del loss
                del outputs

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                writer.writerow({'mAP': str(f'{last[1]:.6f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'mAP@50': str(f'{last[0]:.6f}')})
                f.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                ckpt = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema).half()
                        }

                # Save last, best and delete
                # torch.save(ckpt, './weights/last.pt')
                # if best == last[1]:
                #     torch.save(ckpt, './weights/best.pt')
                # del ckpt
                torch.save(ckpt, f=f'./weights/last_{version}_{args.epochs}.pt')
                # if best == last[0]:
                if best == last[1]:
                    torch.save(ckpt, f=f'./weights/best_{version}_{args.epochs}.pt')
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer(f'./weights/best_{version}_{args.epochs}.pt')  # strip optimizers
        util.strip_optimizer(f'./weights/last_{version}_{args.epochs}.pt')  # strip optimizers

    torch.cuda.empty_cache()


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
    loader = data.DataLoader(dataset, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load(f'./weights/best_{version}_{epochs}.pt', map_location='cuda', weights_only=False)['model'].float()

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
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
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
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return map50, mean_ap

def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    print(f"params amount: {len(params['names'])}")
    version = args.version
    if version == 'n':
        model = nn.yolo_v11_n(len(params['names'])).fuse()
    elif version == 's':
        model = nn.yolo_v11_s(len(params['names'])).fuse()
    elif version == 'm':
        model = nn.yolo_v11_m(len(params['names'])).fuse()
    elif version == 'l':
        model = nn.yolo_v11_l(len(params['names'])).fuse()
    elif version == 'x':
        model = nn.yolo_v11_x(len(params['names'])).fuse()
    else:
        raise ValueError(f"Unsupported YOLOv11 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")
    # model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')

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
        x = torch.from_numpy(x)
        x = x.unsqueeze(dim=0)
        x = x.cuda()
        x = x.half()
        x = x / 255

        # Inference
        outputs = model(x)
        # NMS
        outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
        
        # End timing and calculate latency
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        
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
        latency_text = f"Latency: {latency_ms:.2f} ms"
        cv2.putText(frame, latency_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Inference Result', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # The existing code for video and camera inference (which works)
        # This part remains unchanged
        # model = model['model'].float()
        # model = torch.load(f'./weights/best_{args.version}_{args.epochs}.pt', 'cuda', weights_only=False)['model'].float()
        model.half()
        model.eval()

        if source_type == "video":
            camera = cv2.VideoCapture('src/crowd1.mp4')
        elif source_type == "camera":
            camera = cv2.VideoCapture(0)

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
                x = torch.from_numpy(x)
                x = x.unsqueeze(dim=0)
                x = x.cuda()
                x = x.half()
                x = x / 255
                outputs = model(x)
                outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
                end_event.record()
                torch.cuda.synchronize()
                latency_ms = start_event.elapsed_time(end_event)
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
                
                fps_text = f"FPS: {fps_display:.2f}"
                latency_text = f"Latency: {latency_ms:.2f} ms"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Frame', frame)
                out.write(frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        camera.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--version', default='m', type=str)
    parser.add_argument('--zip', action='store_true')
    parser.add_argument("--inference", type=str, choices=["image", "video", "camera"])

    args = parser.parse_args()
    print(args)

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)
    util.setup_seed()
    util.setup_multi_processes()
    profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)

    if args.inference:
        # print(args.inference)
        # version = args.version
        # if version == 'n':
        #     model = nn.yolo_v11_n(len(params['names']))
        # elif version == 's':
        #     model = nn.yolo_v11_s(len(params['names']))
        # elif version == 'm':
        #     model = nn.yolo_v11_m(len(params['names']))
        # elif version == 'l':
        #     model = nn.yolo_v11_l(len(params['names']))
        # elif version == 'x':
        #     model = nn.yolo_v11_x(len(params['names']))
        # else:
        #     raise ValueError(f"Unsupported YOLOv11 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")
        # model_path = f"./weights/original/best.pt"
        model_path = f"./weights/best_{args.version}_{args.epochs}.pt"
        model_data = torch.load(model_path, map_location="cuda", weights_only=False)
        model = model_data["model"].eval().cuda()
        inference(model, args, params)

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
