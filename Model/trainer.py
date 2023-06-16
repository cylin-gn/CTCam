class Trainer():
    def __init__(self, model,model_2, model_3, lrate, wdecay, clip, step_size, seq_out_len, scaler, scaler_2, device, cl=True):

        # GCT
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        # CCTV
        self.scaler_2 = scaler_2
        self.model_2 = model_2
        self.model_2.to(device)
        self.optimizer_2 = optim.Adam(self.model_2.parameters(), lr=lrate, weight_decay=wdecay)

        # Fusion
        self.model_3 = model_3
        self.model_3.to(device)
        self.optimizer_3 = optim.Adam(self.model_3.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    # input: GCT, input_2: CCTV, real_val: gct
    def train(self, input,input_2, real_val, idx=None):
        self.model.eval()
        self.model_2.eval()

        output = self.model(input)
        output = output.transpose(1,3)
        output_2 = self.model_2(input_2)
        output_2 = output_2.transpose(1,3)

        output = torch.cat([output,input[:,1].unsqueeze(1)],dim=1)
        output_2 = torch.cat([output_2,input_2[:,1].unsqueeze(1)],dim=1)

        #--------------------------------------------#
        self.model_3.train()  #*******#
        self.optimizer_3.zero_grad()  #*******#
        output = self.model_3(output, idx=idx, input_2=output_2) #*******#
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)

        predict = self.scaler.inverse_transform(output)

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
            print("### cl learning\n iter",self.iter,"\niter%step",self.iter%self.step,"\ntask_level",self.task_level)
            print("# predict len:", len(predict[:, :, :, :self.task_level]))

        if self.cl:
            loss = masked_mae(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = masked_mae(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model_3.parameters(), self.clip)  #*******#

        self.optimizer_3.step() #*******#


        mae = masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        smape = masked_smape(predict,real,0.0).item()
        self.iter += 1
        return mae,mape,rmse,smape

    def eval(self, model_type, input, real_val, input_2=None):

        if model_type == 'gct':
          self.model.eval()
          output = self.model(input)
          output = output.transpose(1,3)

          predict = self.scaler.inverse_transform(output)

        elif model_type == 'cctv':
          self.model_2.eval()
          output = self.model_2(input)
          output = output.transpose(1,3)

          predict = self.scaler_2.inverse_transform(output)

        elif model_type == 'fusion':
          self.model.eval()
          self.model_2.eval()
          self.model_3.eval()

          output = self.model(input)
          output = output.transpose(1,3)
          output_2 = self.model_2(input_2)
          output_2 = output_2.transpose(1,3)

          output = torch.cat([output,input[:,1].unsqueeze(1)],dim=1)
          output_2 = torch.cat([output_2,input_2[:,1].unsqueeze(1)],dim=1)

          #print('output', output[0,0,0])
          #print('output_2', output_2[0,0,0])
          output = self.model_3(output, idx=None, input_2=output_2)
          output = output.transpose(1,3)

          predict = self.scaler.inverse_transform(output)

        real = torch.unsqueeze(real_val,dim=1)



        loss = self.loss(predict, real, 0.0)
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        smape = masked_smape(predict,real,0.0).item()
        return loss.item(),mape,rmse,smape

