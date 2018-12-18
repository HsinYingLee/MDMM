import networks
import torch
import torch.nn as nn

class MD_uni(nn.Module):
  def __init__(self, opts):
    super(MD_uni, self).__init__()
    self.opts = opts
    lr = 0.0001
    lr_dcontent = lr/2.5 

    self.isDcontent = opts.isDcontent
  
    self.dis = networks.MD_Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=3)
    self.enc_c = networks.MD_E_content(opts.input_dim_a)
    self.gen = networks.MD_G_uni(opts.input_dim_a, c_dim=3)
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    #if self.isDcontent:
    self.disContent = networks.MD_Dis_content() 
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)

    self.cls_loss = nn.BCEWithLogitsLoss()

  def initialize(self):
    self.dis.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.dis_sch = networks.get_scheduler(self.dis_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.dis_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.dis.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    if self.isDcontent:
      self.disContent.cuda(self.gpu)

  def forward(self):
    # input images
    self.real_img = self.input
    c_org = self.c_org
    c_trg = self.c_trg

    # get encoded z_c
    self.z_content = self.enc_c.forward(self.real_img)

    # reconstruct
    self.fake_recon = self.gen(self.z_content, c_org)

    # translation
    self.fake_img = self.gen(self.z_content, c_trg)

    # for display
    self.image_display = torch.cat((self.real_img[0:1].detach().cpu(), self.fake_recon[0:1].detach().cpu(), \
                                    self.fake_img[0:1].detach().cpu()), dim=0)


  def update_D_content(self, image, c_org):
    self.input = image
    self.z_content = self.enc_c.forward(self.input)
    self.disContent_opt.zero_grad()
    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.cls_loss(pred_cls, c_org)
    loss_D_content.backward()
    self.disContent_loss = loss_D_content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image, c_org, c_trg):
    self.input = image
    self.c_org = c_org
    self.c_trg = c_trg
    self.forward()

    self.dis_opt.zero_grad()
    self.D_gan_loss, self.D_cls_loss, self.D_gp_loss = self.backward_D(self.dis, self.real_img, self.fake_img)
    self.dis_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake, pred_fake_cls = netD.forward(fake.detach())
    pred_real, pred_real_cls = netD.forward(real)
    '''
    loss_D_gan = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D_gan += ad_true_loss + ad_fake_loss
    '''
    loss_D_gan = torch.mean(pred_fake) - torch.mean(pred_real)    
    alpha = torch.rand(real.size(0), 1, 1, 1).to(self.gpu)
    x_hat = (alpha * real.data + (1 - alpha) * fake.data).requires_grad_(True)
    out_src, _ = netD(x_hat)
    loss_D_gp = self.gradient_penalty(out_src, x_hat)


    loss_D_cls = self.cls_loss(pred_real_cls, self.c_org)
    loss_D = loss_D_gan + self.opts.lambda_cls * loss_D_cls  + 10*loss_D_gp
    loss_D.backward()
    return loss_D_gan, loss_D_cls, loss_D_gp

  def update_EG(self):
    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    if self.opts.isDcontent:
      loss_G_GAN_content = self.backward_G_GAN_content(self.z_content)


    # Ladv for generator
    pred_fake, pred_fake_cls = self.dis.forward(self.fake_img)
    '''
    loss_G_GAN = 0
    for  ut_a in pred_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G_GAN += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    '''
    loss_G_GAN = -torch.mean(pred_fake)
    # classification
    loss_G_cls = self.cls_loss(pred_fake_cls, self.c_trg) * self.opts.lambda_cls

    # recon
    loss_G_rec = torch.mean(torch.abs(self.input - self.fake_recon)) * self.opts.lambda_rec


    # KL loss - z_c
    loss_kl_zc = self._l2_regularize(self.z_content) * 0.01 *0


    loss_G = loss_G_GAN + loss_G_cls + loss_G_rec + loss_kl_zc
    if self.opts.isDcontent:
      loss_G += loss_G_GAN_content
    loss_G.backward(retain_graph=True)

    self.gan_loss = loss_G_GAN.item()
    self.gan_cls_loss = loss_G_cls.item()
    if self.opts.isDcontent:
      self.gan_loss_content = loss_G_GAN_content.item()
    self.kl_loss_zc = loss_kl_zc.item()
    self.l1_rec_loss = loss_G_rec.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    pred_cls = self.disContent.forward(data)
    loss_G_content = self.cls_loss(pred_cls, 1-self.c_org)
    return loss_G_content

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def gradient_penalty(self, y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(self.gpu)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

  def update_lr(self):
    self.dis_sch.step()
    if self.opts.isDcontent:
      self.disContent_sch.step()
    self.enc_c_sch.step()
    self.gen_sch.step()

  def assemble_outputs(self):
    images_real = self.normalize_image(self.real_img).detach()
    images_fake_recon = self.normalize_image(self.fake_recon).detach()
    images_fake_img = self.normalize_image(self.fake_img).detach()
    row1 = torch.cat((images_real[0:1, ::], images_fake_recon[0:1, ::], images_fake_img[0:1, ::]),3)
    return row1

  def normalize_image(self, x):
    return x[:,0:3,:,:]


  def save(self, filename, ep, total_it):
    state = {
             'dis': self.dis.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'gen': self.gen.state_dict(),
             'dis_opt': self.dis_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.dis.load_state_dict(checkpoint['dis'])
      if self.isDcontent:
        self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.dis_opt.load_state_dict(checkpoint['dis_opt'])
      if self.isDcontent:
        self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

############################################################################################################

class MD_multi(nn.Module):
  def __init__(self, opts):
    super(MD_multi, self).__init__()
    self.opts = opts
    lr = 0.0001
    lr_dcontent = lr/2.5 
    self.nz = 8

    self.isDcontent = opts.isDcontent
    if opts.concat == 1:
      self.concat = True
    else:
      self.concat = False
  
    self.dis1 = networks.MD_Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=3)
    self.dis2 = networks.MD_Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm, c_dim=3)
    self.enc_c = networks.MD_E_content(opts.input_dim_a)
    if self.concat:
      self.enc_a = networks.MD_E_attr_concat(opts.input_dim_a, output_nc=self.nz, c_dim=3, \
          norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
      self.gen = networks.MD_G_multi_concat(opts.input_dim_a, c_dim=3, nz=self.nz)
    else:
      self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)
      self.gen = networks.MD_G_multi(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    self.dis1_opt = torch.optim.Adam(self.dis1.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis2_opt = torch.optim.Adam(self.dis2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    #if self.isDcontent:
    self.disContent = networks.MD_Dis_content() 
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)

    self.cls_loss = nn.BCEWithLogitsLoss()

  def initialize(self):
    self.dis1.apply(networks.gaussian_weights_init)
    self.dis2.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.dis1_sch = networks.get_scheduler(self.dis1_opt, opts, last_ep)
    self.dis2_sch = networks.get_scheduler(self.dis2_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def update_lr(self):
    self.dis1_sch.step()
    self.dis2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.gen_sch.step()

  def setgpu(self, gpu):
    self.gpu = gpu
    self.dis1.cuda(self.gpu)
    self.dis2.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    if self.isDcontent:
      self.disContent.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def forward(self):
    # input images
    if not self.input.size(0)%2 == 0:
      print("Need to be even QAQ")
      input()
    half_size = self.input.size(0)//2
    self.real_A = self.input[0:half_size]
    self.real_B = self.input[half_size:]
    c_org_A = self.c_org[0:half_size]
    c_org_B = self.c_org[half_size:]
    c_trg_a = self.c_trg

    # get encoded z_c
    self.real_img = torch.cat((self.real_A, self.real_B),0)
    self.z_content = self.enc_c.forward(self.real_img)
    self.z_content_a, self.z_content_b = torch.split(self.z_content, half_size, dim=0)

    # get encoded z_a
    if self.concat:
      self.mu, self.logvar = self.enc_a.forward(self.real_img, self.c_org)
      std = self.logvar.mul(0.5).exp_()
      eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
      self.z_attr = eps.mul(std).add_(self.mu)
    else:
      self.z_attr = self.enc_a.forward(self.real_img, self.c_org)
    self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, half_size, dim=0)
    # get random z_a
    self.z_random = self.get_z_random(half_size, self.nz, 'gauss')

    # first cross translation
    input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b),0)
    input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a),0)
    input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
    input_c_forA = torch.cat((c_org_A, c_org_A, c_org_A), 0)
    input_c_forB = torch.cat((c_org_B, c_org_B, c_org_B), 0)
    output_fakeA = self.gen.forward(input_content_forA, input_attr_forA, input_c_forA)
    output_fakeB = self.gen.forward(input_content_forB, input_attr_forB, input_c_forB)
    self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get reconstructed encoded z_c
    self.fake_encoded_img = torch.cat((self.fake_A_encoded, self.fake_B_encoded),0)
    self.z_content_recon = self.enc_c.forward(self.fake_encoded_img)
    self.z_content_recon_b, self.z_content_recon_a = torch.split(self.z_content_recon, half_size, dim=0)

    # get reconstructed encoded z_a
    if self.concat:
      self.mu_recon, self.logvar_recon = self.enc_a.forward(self.fake_encoded_img, self.c_org)
      std_recon = self.logvar_recon.mul(0.5).exp_()
      eps_recon = self.get_z_random(std_recon.size(0), std_recon.size(1), 'gauss')
      self.z_attr_recon = eps.mul(std_recon).add_(self.mu_recon)
    else:
      self.z_attr_recon = self.enc_a.forward(self.fake_encoded_img, self.c_org)
    self.z_attr_recon_a, self.z_attr_recon_b = torch.split(self.z_attr_recon, half_size, dim=0)

    # second cross translation
    self.fake_A_recon = self.gen.forward(self.z_content_recon_a, self.z_attr_recon_a, c_org_A)
    self.fake_B_recon = self.gen.forward(self.z_content_recon_b, self.z_attr_recon_b, c_org_B)

    # for display
    self.image_display = torch.cat((self.real_A[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                    self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(), \
                                    self.real_B[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                    self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

    # for latent regression
    self.fake_random_img = torch.cat((self.fake_A_random, self.fake_B_random), 0)
    if self.concat:
      self.mu2, _= self.enc_a.forward(self.fake_random_img, self.c_org)
      self.mu2_a, self.mu2_b = torch.split(self.mu2, half_size, 0)
    else:
      self.z_attr_random = self.enc_a.forward(self.fake_random_img, self.c_org)
      self.z_attr_random_a, self.z_attr_random_b = self.split(self.attr_random, half_size, 0)


  def update_D_content(self, image, c_org):
    self.input = image
    self.z_content = self.enc_c.forward(self.input)
    self.disContent_opt.zero_grad()
    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.cls_loss(pred_cls, c_org)
    loss_D_content.backward()
    self.disContent_loss = loss_D_content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image, c_org, c_trg):
    self.input = image
    self.c_org = c_org
    self.c_trg = c_trg
    self.forward()

    self.dis1_opt.zero_grad()
    self.D1_gan_loss, self.D1_cls_loss = self.backward_D(self.dis1, self.input, self.fake_encoded_img)
    self.dis1_opt.step()

    self.dis2_opt.zero_grad()
    self.D2_gan_loss, self.D2_cls_loss = self.backward_D(self.dis2, self.input, self.fake_random_img)
    self.dis2_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake, pred_fake_cls = netD.forward(fake.detach())
    pred_real, pred_real_cls = netD.forward(real)
    loss_D_gan = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D_gan += ad_true_loss + ad_fake_loss
    loss_D_cls = self.cls_loss(pred_real_cls, self.c_org)
    loss_D = loss_D_gan + self.opts.lambda_cls * loss_D_cls 
    loss_D.backward()
    return loss_D_gan, loss_D_cls

  def update_EG(self):
    # update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    if self.opts.isDcontent:
      loss_G_GAN_content = self.backward_G_GAN_content(self.z_content)

    # Ladv for generator
    pred_fake, pred_fake_cls = self.dis1.forward(self.fake_encoded_img)
    loss_G_GAN = 0
    for out_a in pred_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G_GAN += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
   
    # classification
    loss_G_cls = self.cls_loss(pred_fake_cls, self.c_org) * self.opts.lambda_cls

    # self and cross-cycle recon
    loss_G_L1_self = torch.mean(torch.abs(self.input - torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0))) * self.opts.lambda_rec
    loss_G_L1_cc = torch.mean(torch.abs(self.input - torch.cat((self.fake_A_recon, self.fake_B_recon), 0))) * self.opts.lambda_rec


    # KL loss - z_c
    loss_kl_zc = self._l2_regularize(self.z_content) * 0.01 * 0

    # KL loss - z_a
    if self.concat:
      kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
      loss_kl_za = torch.sum(kl_element).mul_(-0.5) * 0.01
    else:
      loss_kl_za = self._l2_regularize(self.z_attr) * 0.01


    loss_G = loss_G_GAN + loss_G_cls + loss_G_L1_self + loss_G_L1_cc + loss_kl_zc + loss_kl_za
    if self.opts.isDcontent:
      loss_G += loss_G_GAN_content
    loss_G.backward(retain_graph=True)

    self.gan_loss = loss_G_GAN.item()
    self.gan_cls_loss = loss_G_cls.item()
    if self.opts.isDcontent:
      self.gan_loss_content = loss_G_GAN_content.item()
    self.kl_loss_zc = loss_kl_zc.item()
    self.kl_loss_za = loss_kl_za.item()
    self.l1_self_rec_loss = loss_G_L1_self.item()
    self.l1_cc_rec_loss = loss_G_L1_cc.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    pred_cls = self.disContent.forward(data)
    loss_G_content = self.cls_loss(pred_cls, 1-self.c_org)
    return loss_G_content

  def backward_G_alone(self):
    # Ladv for generator
    pred_fake, pred_fake_cls = self.dis2.forward(self.fake_random_img)
    loss_G_GAN2 = 0
    for out_a in pred_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G_GAN2 += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

    # latent regression loss
    if self.concat:
      loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    else:
      loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

    loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2
    loss_z_L1.backward()
    self.l1_recon_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
    self.gan2_loss = loss_G_GAN2.item()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def update_lr(self):
    self.dis1_sch.step()
    self.dis2_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A).detach()
    images_b = self.normalize_image(self.real_B).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a2 = self.normalize_image(self.fake_A_random).detach()
    images_a3 = self.normalize_image(self.fake_A_recon).detach()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]


  def save(self, filename, ep, total_it):
    state = {
             'dis1': self.dis1.state_dict(),
             'dis2': self.dis2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'dis1_opt': self.dis1_opt.state_dict(),
             'dis2_opt': self.dis2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.dis1.load_state_dict(checkpoint['dis1'])
      self.dis2.load_state_dict(checkpoint['dis2'])
      if self.isDcontent:
        self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.dis1_opt.load_state_dict(checkpoint['dis1_opt'])
      self.dis2_opt.load_state_dict(checkpoint['dis2_opt'])
      if self.isDcontent:
        self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

class DRIT(nn.Module):
  def __init__(self, opts):
    super(DRIT, self).__init__()

    # parameters
    lr = 0.0001
    lr_dcontent = lr / 2.5
    self.nz = 8
    self.concat = opts.concat

    # discriminators
    if opts.dis_scale > 1:
      self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    self.disContent = networks.Dis_content()

    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)
    if self.concat:
      self.enc_a = networks.E_attr_concat(opts.input_dim_a, opts.input_dim_b, self.nz, \
          norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
    else:
      self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

    # generator
    if self.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()

  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.disA2.apply(networks.gaussian_weights_init)
    self.disB2.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.disA2_sch = networks.get_scheduler(self.disA2_opt, opts, last_ep)
    self.disB2_sch = networks.get_scheduler(self.disB2_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.disA2.cuda(self.gpu)
    self.disB2.cuda(self.gpu)
    self.disContent.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def test_forward(self, image, a2b=True):
    self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    if a2b:
        self.z_content = self.enc_c.forward_a(image)
        output = self.gen.forward_b(self.z_content, self.z_random)
    else:
        self.z_content = self.enc_c.forward_b(image)
        output = self.gen.forward_a(self.z_content, self.z_random)
    return output

  def test_forward_transfer(self, image_a, image_b, a2b=True):
    self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
    if a2b:
      output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
    else:
      output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
    return output

  def forward(self):
    # input images
    half_size = 1
    real_A = self.input_A
    real_B = self.input_B
    self.real_A_encoded = real_A[0:half_size]
    self.real_A_random = real_A[half_size:]
    self.real_B_encoded = real_B[0:half_size]
    self.real_B_random = real_B[half_size:]

    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

    # get encoded z_a
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)

    # get random z_a
    self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

    # first cross translation
    input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b),0)
    input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a),0)
    input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
    output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
    output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
    self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get reconstructed encoded z_c
    self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded, self.fake_B_encoded)

    # get reconstructed encoded z_a
    if self.concat:
      self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
      std_a = self.logvar_recon_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
      std_b = self.logvar_recon_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
    else:
      self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)

    # second cross translation
    self.fake_A_recon = self.gen.forward_a(self.z_content_recon_a, self.z_attr_recon_a)
    self.fake_B_recon = self.gen.forward_b(self.z_content_recon_b, self.z_attr_recon_b)

    # for display
    self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                    self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(), \
                                    self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                    self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

    # for latent regression
    if self.concat:
      self.mu2_a, _, self.mu2_b, _ = self.enc_a.forward(self.fake_A_random, self.fake_B_random)
    else:
      self.z_attr_random_a, self.z_attr_random_b = self.enc_a.forward(self.fake_A_random, self.fake_B_random)

  def forward_content(self):
    half_size = 1
    self.real_A_encoded = self.input_A[0:half_size]
    self.real_B_encoded = self.input_B[0:half_size]
    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

  def update_D_content(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward_content()
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disA2
    self.disA2_opt.zero_grad()
    loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
    self.disA2_loss = loss_D2_A.item()
    self.disA2_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()

    # update disB2
    self.disB2_opt.zero_grad()
    loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
    self.disB2_loss = loss_D2_B.item()
    self.disB2_opt.step()

    # update disContent
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def backward_contentD(self, imageA, imageB):
    pred_fake = self.disContent.forward(imageA.detach())
    pred_real = self.disContent.forward(imageB.detach())
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
      all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
    loss_D = ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
    loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

    # Ladv for generator
    loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

    # KL loss - z_a
    if self.concat:
      kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
      loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
      kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
      loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
    else:
      loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
      loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

    # KL loss - z_c
    loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
    loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01

    # cross cycle consistency loss
    loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

    loss_G = loss_G_GAN_A + loss_G_GAN_B + \
             loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
             loss_G_L1_AA + loss_G_L1_BB + \
             loss_G_L1_A + loss_G_L1_B + \
             loss_kl_zc_a + loss_kl_zc_b + \
             loss_kl_za_a + loss_kl_za_b

    loss_G.backward(retain_graph=True)

    self.gan_loss_a = loss_G_GAN_A.item()
    self.gan_loss_b = loss_G_GAN_B.item()
    self.gan_loss_acontent = loss_G_GAN_Acontent.item()
    self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
    self.kl_loss_za_a = loss_kl_za_a.item()
    self.kl_loss_za_b = loss_kl_za_b.item()
    self.kl_loss_zc_a = loss_kl_zc_a.item()
    self.kl_loss_zc_b = loss_kl_zc_b.item()
    self.l1_recon_A_loss = loss_G_L1_A.item()
    self.l1_recon_B_loss = loss_G_L1_B.item()
    self.l1_recon_AA_loss = loss_G_L1_AA.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    outs = self.disContent.forward(data)
    for out in outs:
      outputs_fake = nn.functional.sigmoid(out)
      all_half = 0.5*torch.ones((outputs_fake.size(0))).cuda(self.gpu)
      ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
    return ad_loss

  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self):
    # Ladv for generator
    loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disA2)
    loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)

    # latent regression loss
    if self.concat:
      loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    else:
      loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

    loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
    loss_z_L1.backward()
    self.l1_recon_z_loss_a = loss_z_L1_a.item()
    self.l1_recon_z_loss_b = loss_z_L1_b.item()
    self.gan2_loss_a = loss_G_GAN2_A.item()
    self.gan2_loss_b = loss_G_GAN2_B.item()

  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.disA2_sch.step()
    self.disB2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disA2.load_state_dict(checkpoint['disA2'])
      self.disB.load_state_dict(checkpoint['disB'])
      self.disB2.load_state_dict(checkpoint['disB2'])
      self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
      self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'disA': self.disA.state_dict(),
             'disA2': self.disA2.state_dict(),
             'disB': self.disB.state_dict(),
             'disB2': self.disB2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disA2_opt': self.disA2_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disB2_opt': self.disB2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a2 = self.normalize_image(self.fake_A_random).detach()
    images_a3 = self.normalize_image(self.fake_A_recon).detach()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
