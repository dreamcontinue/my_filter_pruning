import torch
import numpy as np
import utils

# layer_end_for_different_structure()
# {
# resnet110 324
# resnet56 162
# resnet32 90
# resnet20 54
# }
class Mask:
    def __init__(self, model,rate,last_index):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.rate=rate
        self.last_index=last_index
        self.init_length()
        self.init_rate(self.rate,self.last_index)

    # 计算conv对应的mask
    def get_sfp_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

        else:
            pass
        # print("filter codebook done")
        # return codebook
        return codebook.reshape(weight_torch.size())


    # 计算conv对应的mask
    def get_filter_codebook(self, weight_torch, compress_rate, length):
        # return self.get_sfp_filter_codebook(weight_torch, compress_rate, length)
        codebook_1 = np.ones(length)
        codebook_2 = np.ones(length)
        if len(weight_torch.size()) == 4:
            # row
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook_1[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
            codebook_1=codebook_1.reshape(weight_torch.size())

            # col
            # transpose first and then same with row
            weight_torch_transposed = weight_torch.transpose(1, 0).contiguous()# pytorch transpose
            filter_pruned_num = int(weight_torch_transposed.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch_transposed.view(weight_torch_transposed.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch_transposed.size()[1] * weight_torch_transposed.size()[2] * \
                            weight_torch_transposed.size()[3]

            for x in range(0, len(filter_index)):
                codebook_2[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
            # at last transposed back
            codebook_2 = codebook_2.reshape(weight_torch_transposed.size())
            codebook_2 = codebook_2.transpose(1, 0, 2, 3)# numpy transpose
            # codebook_2 = codebook_2.flatten()
        else:
            pass
        # print_mask(codebook_1*codebook_2)
        return codebook_1 * codebook_2

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    # 初始化每一层的参数长度
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
        #
        # print(self.model_size)
        # print(self.model_length)
        #


    # 初始化网络的压缩比率和要压缩层的index
    def init_rate(self, layer_rate,last_index):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            # simply mask all conv
            if len(item.data.size())==4 and index<last_index:
                self.mask_index.append(index)
                self.compress_rate[index] = layer_rate
        #
        # print(self.mask_index)
        #

    # 计算当前网络的mask
    def init_mask(self, use_cuda=True):
        # 初始化横向纵向mask
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
        print('init mask')
        self.print_masks_zero()
        # mask前向后向更新
        # 后面竖的与前面横的对应（?,a,?,?）(a,?,?,?)
        pre_index=self.mask_index[0]
        for index in self.mask_index[1:]:
            # index 后面的序号 pre_index 前面的序号
            filter_num=self.mat[index].shape[1]
            for i in range(filter_num):
                filter_i_sum=np.sum(self.mat[index][:,i,:,:])
                if filter_i_sum==0:
                    self.mat[pre_index][i,:,:,:]=0
            pre_index=index
        print('from rear to head')
        self.print_masks_zero()

        # 前面横的与后面竖的对应（a,?,?,?）(?,a,?,?)
        post_index=self.mask_index[-1]
        for index in reversed(self.mask_index[:-1]):
            # index 前面的序号 post_index 后面的序号
            filter_num=self.mat[index].shape[0]
            for i in range(filter_num):
                filter_i_sum=np.sum(self.mat[index][i,:,:,:])
                if filter_i_sum==0:
                    self.mat[post_index][:,i,:,:]=0
            post_index=index
        print('from head to rear')
        self.print_masks_zero()

        # numpy to tensor
        for index in self.mask_index:
            self.mat[index] = self.convert2tensor(self.mat[index])
            if use_cuda:
                self.mat[index] = self.mat[index].cuda()
        print("mask Ready")


    # 进行mask
    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                item.data=item.data*self.mat[index]
        print("mask Done")

    # 输出0的数量
    def if_zero(self):
        non_zeros=utils.AverageMeter()
        zeros=utils.AverageMeter()
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if(index ==0):
                a = item.data
                b = a.cpu().numpy()
                non_zeros.update(np.count_nonzero(b))
                zeros.update(b.size-np.count_nonzero(b))

                # print("number of nonzero weight is %d, zero  is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
        print("number of nonzero weight is %d, zero  is %d" % (non_zeros.sum,zeros.sum))

    def print_masks_zero(self):
        non_zeros=utils.AverageMeter()
        zeros=utils.AverageMeter()
        for index in self.mask_index:
            a=self.mat[index]
            non_zeros.update(np.count_nonzero(a))
            zeros.update(a.size-np.count_nonzero(a))
        print("number of nonzero weight is %d, zero  is %d" % (non_zeros.sum,zeros.sum))

def print_mask(mask):
    h,w=mask.shape[0:2]
    codebook=np.zeros([h,w],dtype=int)
    for i in range(h):
        for j in range(w):
            codebook[i,j]=np.count_nonzero(mask[i,j])
    np.set_printoptions(threshold=1000)
    print(codebook)

def print_masks_zeros(mask):
    non_zeros=np.count_nonzero(mask)
    zeros=mask.size-non_zeros
    print("number of nonzero weight is %d, zero  is %d" % (non_zeros, zeros))

