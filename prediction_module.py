import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden_dim', type=int, default=32, help='number of hidden dim for ConvLSTM layers')
opt = parser.parse_args()


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # print(combined.size())
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_3_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_3_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4,
                    h_t5, c_t5, h_t6, c_t6):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            h_t3, c_t3 = self.encoder_2_convlstm(input_tensor=h_t2,
                                                 cur_state=[h_t3, c_t3])

        # encoder_vector
        encoder_vector = h_t3

        # decoder
        for t in range(future_step):
            h_t4, c_t4 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            h_t5, c_t5 = self.decoder_2_convlstm(input_tensor=h_t4,
                                                 cur_state=[h_t5, c_t5])  # we could concat to provide skip conn here
            h_t6, c_t6 = self.decoder_2_convlstm(input_tensor=h_t5,
                                                 cur_state=[h_t6, c_t6])
            encoder_vector = h_t6
            outputs += [h_t6]  # predictions

        # print(outputs)
        outputs = torch.stack(outputs, 1)
        # print(outputs.size())
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        # print(outputs.size())
        # outputs = torch.nn.Tanh()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.encoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        h_t4, c_t4 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t5, c_t5 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t6, c_t6 = self.decoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3,
                                   h_t4, c_t4, h_t5, c_t5, h_t6, c_t6)

        return outputs

def train():
    model.train()
    total_loss = 0

    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        # print(features.size())
        # Forward pass
        outputs = model(features, future_step)
        # print('labels size: ', labels.size())
        # print('output size: ', outputs.size())

        # print(images.type())
        # print(outputs.type())
        # print(labels.type())

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * outputs.size()[0]

    return total_loss

def evaluate():
    model.eval()
    total_loss = 0

    for i, (features, labels) in enumerate(val_loader):
        features = features.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(features, future_step)

        # print(images.type())
        # print(outputs.type())
        # print(labels.type())

        loss = criterion(outputs, labels)

        # Backward and optimize

        total_loss += loss.item() * outputs.size()[0]

    return total_loss

class MyDataset(object):

    def __init__(self, data, transform, seq_len=5, num_digits=2, image_size=160, deterministic=True):
        self.seq_len = seq_len
        self.data = data
        self.N = len(self.data) - self.seq_len

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + 1:index + self.seq_len + 1]
        return (x, y)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    epoch_nums = 300
    lr = 1e-3
    future_step = 5
    previous_len = 5
    transform = None

    model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    model = torch.load('prediction.model')

    model.eval()
    test_data = np.load('test_pred_labels.npy')[:, np.newaxis, :, :]
    test_size = test_data.shape[0]
    test_data[test_data==255] = 1
    test_data[test_data == 100] = 0
    print('test shape:', test_data.shape)

    test_data = torch.FloatTensor(test_data)
    test_data = MyDataset(data=test_data, transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    print('output size: ', len(test_data))

    true_l = np.zeros((len(test_data), 160, 160))
    ret_l = np.zeros((len(test_data), 160, 160))
    mse_list = 0

    for i, (f, labels) in enumerate(test_loader):
        features = f.to(device)
        # print(features.size())
        out = model(features, future_step)
        outputs = out.detach().cpu().numpy()
        labels = labels.numpy()
        # print(outputs.shape, labels.shape)

        pred_image = outputs[0][-1][0]

        # print(np.sum(pred_image))
        true_image = labels[0][-1][0]
        # print(np.sum(true_image))
        true_l[i] = true_image
        ret_l[i] = pred_image

    outname = 'detection_prediction_results.npy'
    np.save(file=outname, arr=ret_l)

    #show images
    ret_l = np.load(outname)
    print(ret_l.shape)

    true_data = np.load('test.npy')[previous_len:, -1, :, :]
    part_l = np.load('test.npy')[previous_len:, -2, :, :]
    print(true_data.shape)

    for k in range(len(true_data)):
        true_ret = np.zeros((160, 160))
        pred_ret = np.zeros((160, 160))

        for i in range(160):
            for j in range(160):
                if part_l[k, i, j] > 0:
                    true_i = true_data[k, i, j]
                    pred_i = ret_l[k, i, j]
                    # get true image
                    if true_i > 0:
                        true_ret[i, j] = 255
                    else:
                        true_ret[i, j] = 100

                    # get predict image
                    if pred_i > 0.4:
                        pred_ret[i, j] = 255
                    else:
                        pred_ret[i, j] = 100

        plt.imshow(true_ret, vmin=0, vmax=255)
        plt.title('true' + str(k))
        plt.show()
        plt.imshow(pred_ret, vmin=0, vmax=255)
        plt.title('pred' + str(k))
        plt.show()