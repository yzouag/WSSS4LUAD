import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_small_label(net, train_loader, valid_loader, optimizer, criteria, scheduler, epochs, save_every, setting_str):
    """
    this is the doc

    Args:
        net ([type]): [description]
        train_loader ([type]): [description]
        valid_loader ([type]): [description]
        optimizer ([type]): [description]
        criteria ([type]): [description]
        scheduler ([type]): [description]
        epochs ([type]): [description]
        save_every ([type]): [description]
        setting_str ([type]): [description]
    """
    loss_t = []
    accuracy_t = []
    loss_v = []
    accuracy_v = []

    for i in range(epochs):
        running_loss = 0.
        correct = 0
        net.train()

        for img, label in tqdm(train_loader):
            img = img.cuda()
            label = label.cuda()
            scores = net(img)
            loss = criteria(scores, label.float())
            
            scores = torch.sigmoid(scores)
            predict = torch.zeros_like(scores)
            predict[scores > 0.7] = 1
            predict[scores < 0.3] = 0
            predict[torch.logical_and(scores > 0.3, scores < 0.7)] = -1
            for k in range(len(predict)):
                if torch.equal(predict[k], label[k]):
                    correct += 1

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_acc = correct / float(len(train_loader))
        scheduler.step()

        with torch.no_grad():
            net.eval()
            running_loss = 0.0
            running_corrects = 0
            for img, label in tqdm(valid_loader):
                img = img.cuda()
                label = label.cuda()
                scores = net(img)
                loss = criteria(scores, label.float())
                
                scores = torch.sigmoid(scores)
                predict = torch.zeros_like(scores)
                predict[scores > 0.5] = 1
                predict[scores <= 0.5] = 0
                for k in range(len(predict)):
                    if torch.equal(predict[k], label[k]):
                        correct += 1

            valid_loss = running_loss / len(valid_loader)
            valid_acc = running_corrects / float(len(valid_loader))
        
        print(f'Epoch [{i+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f},  Valid Acc: {valid_acc:.4f}')
        
        accuracy_t.append()
        loss_t.append()
        if (i + 1) % save_every == 0 and (i + 1) != epochs:
            torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + setting_str + "_ep"+str(i+1)+".pth")

    torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./modelstates/" + setting_str + "_last.pth")

    plt.figure(1)
    plt.plot(loss_t)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('./image/train_loss.png')
    plt.close()

    plt.figure(2)
    plt.plot(accuracy_t)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.savefig('./image/train_accuracy.png')

    plt.figure(3)
    plt.plot(loss_v)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('./image/valid_loss.png')
    plt.close()

    plt.figure(4)
    plt.plot(accuracy_v)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.savefig('./image/valid_accuracy.png')