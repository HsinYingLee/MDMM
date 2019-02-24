import torch
from options import TestOptions
from datasets import dataset_single
from model import MD_multi
from saver import save_imgs, save_concat_imgs
import os

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()
  domains = [chr(i) for i in range(ord('A'),ord('Z')+1)]

  # data loader
  print('\n--- load dataset ---')
  datasets = [None]*opts.num_domains
  loaders = [None]*opts.num_domains
  for i in range(opts.num_domains):
    datasets[i] = dataset_single(opts, i)
    loaders[i] = torch.utils.data.DataLoader(datasets[i], batch_size=1, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = MD_multi(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  for d in range(opts.num_domains):
    for idx, data in enumerate(loaders[d]):
      #break
      img, c_org = data
      print('{}/{}'.format(idx, len(loaders[d])))
      if idx > num:
        break;
      img, c_org = img.cuda(), c_org.cuda()
      imgs = [img]
      names = ['input']
      for idx2 in range(opts.num):
        with torch.no_grad():
          imgs_ = model.test_forward_random(img)
        for i in range(opts.num_domains):
          imgs.append(imgs_[i])
          names.append('output{}_{}_{}'.format(domains[d], domains[i], idx2))
      save_imgs(imgs, names, os.path.join(result_dir, '{}_{}'.format(domains[d], idx)))
      save_concat_imgs(imgs, 'output{}_{}'.format(domains[d], idx), result_dir)
  return

if __name__ == '__main__':
  main()
