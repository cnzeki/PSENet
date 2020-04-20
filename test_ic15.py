# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils import data

# c++ version pse based on opencv 3+
from  test_base import *
from torch.utils import data

# c++ version pse based on opencv 3+
from test_base import *
from dataset import IC15TestLoader

def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox['bbox']]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def test(args):
    data_loader = IC15TestLoader(long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    model = load_model(args)
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img) in enumerate(test_loader):
        print(('progress: %d / %d'%(idx, len(test_loader))))
        sys.stdout.flush()

        img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        torch.cuda.synchronize()
        start = time.time()
        bboxes = run_PSENet(args, model, img, org_img.shape, out_type='rbox')
        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print(('fps: %.2f'%(total_frame / total_time)))
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, bboxes, 'outputs/submit_ic15/')

        debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_ic15/')

    cmd = 'cd %s;zip -j %s %s/*'%('./outputs/', 'submit_ic15.zip', 'submit_ic15');
    print(cmd)
    sys.stdout.flush()
    util.cmd.cmd(cmd)


if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()
    test(args)
