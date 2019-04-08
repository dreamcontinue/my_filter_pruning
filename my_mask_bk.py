import torch
import numpy as np
import utils

class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.init_length()

    # def get_codebook(self, weight_torch, compress_rate, length):
    #     weight_vec = weight_torch.view(length)
    #     weight_np = weight_vec.cpu().numpy()
    #
    #     weight_abs = np.abs(weight_np)
    #     weight_sort = np.sort(weight_abs)
    #
    #     threshold = weight_sort[int(length * (1 - compress_rate))]
    #     weight_np[weight_np <= -threshold] = 1
    #     weight_np[weight_np >= threshold] = 1
    #     weight_np[weight_np != 1] = 0
    #
    #     print("codebook done")
    #     return weight_np

    # 计算conv对应的mask
    def get_sfp_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

        else:
            pass
        # print("filter codebook done")
        return codebook

    # 计算conv对应的mask
    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook_1 = np.ones(length)
        codebook_2 = np.ones(length)
        if len(weight_torch.size()) == 4:
            # row
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook_1[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
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
            codebook_2 = codebook_2.reshape([weight_torch.size()[1],
                                             weight_torch.size()[0],
                                             weight_torch.size()[2],
                                             weight_torch.size()[3]])
            codebook_2 = codebook_2.transpose(1, 0, 2, 3)# numpy transpose
            codebook_2 = codebook_2.flatten()
        else:
            pass
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

    # layer_end_for_different_structure()
    # {
    # resnet110 324
    # resnet56 162
    # resnet32 90
    # resnet20 54
    # }

    # 初始化网络的压缩比率和要压缩层的index
    def init_rate(self, layer_rate,last_index):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            # simply mask all conv
            if len(item.data.size())==4 and index<last_index:
                self.mask_index.append(index)
                self.compress_rate[index] = layer_rate

        # print(self.compress_rate)
        # print(self.mask_index)
        # for key in range(layer_begin, layer_end + 1, layer_inter):
        #     self.compress_rate[key] = layer_rate
        # self.mask_index = [x for x in range(0, last_index, conv_inter)]
        #
        # print(self.compress_rate)
        # print(self.mask_index)

    # def init_rate(self, layer_rate,layer_begin,layer_end,layer_inter,last_index,conv_inter):
    #     for index, item in enumerate(self.model.parameters()):
    #         self.compress_rate[index] = 1
    #     for key in range(layer_begin, layer_end + 1, layer_inter):
    #         self.compress_rate[key] = layer_rate
    #     self.mask_index = [x for x in range(0, last_index, conv_inter)]
    #
    #     print(self.compress_rate)
    #     print(self.mask_index)

    # def init_rate(self, layer_rate):
    #     for index, item in enumerate(self.model.parameters()):
    #         self.compress_rate[index] = 1
    #     for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
    #         self.compress_rate[key] = layer_rate
    #     # different setting for  different architecture
    #     if args.arch == 'resnet20':
    #         last_index = 57
    #     elif args.arch == 'resnet32':
    #         last_index = 93
    #     elif args.arch == 'resnet56':
    #         last_index = 165
    #     elif args.arch == 'resnet110':
    #         last_index = 327
    #     self.mask_index = [x for x in range(0, last_index, 3)]

    # 计算当前网络的mask
    def init_mask(self, layer_rate,  last_index, use_cuda=True):
        self.init_rate(layer_rate,last_index)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if use_cuda:
                    self.mat[index] = self.mat[index].cuda()
        # print("mask Ready")
        # print(self.mat)

    # 进行mask
    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        # print("mask Done")

    # 输出0的数量
    def if_zero(self):
        non_zeros=utils.AverageMeter()
        zeros=utils.AverageMeter()
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if(index ==0):
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                non_zeros.update(np.count_nonzero(b))
                zeros.update(len(b)-np.count_nonzero(b))

                # print("number of nonzero weight is %d, zero  is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
        print("number of nonzero weight is %d, zero  is %d" % (non_zeros.sum,zeros.sum))
