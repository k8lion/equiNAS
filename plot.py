import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    file='logs2022-05-13_11:53:01'
    with open('out/'+file+'.pkl','rb') as savefile:
        save = pickle.load(savefile)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,10))
    ax1.plot(save['steerable']['train']['loss'], label='C8 Steerable CNN')
    ax1.plot(save['unsteerable']['train']['loss'], label='Vanilla CNN (equal channels)')
    ax1.legend()
    ax1.set_title('Loss')
    ax1.set_yscale('log')
    ax2.plot(save['steerable']['validation']['accuracy'], label='C8 Steerable')
    ax2.plot(save['unsteerable']['validation']['accuracy'], label='Vanilla CNN (equal channels)')
    ax2.legend()
    ax2.set_title('Accuracy')
    plt.savefig('out/'+file+'.png')
