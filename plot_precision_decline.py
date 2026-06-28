import matplotlib.pyplot as plt

def plot_precision_decline(img2txt_precision,txt2img_precision):
    plt.figure(figsize=(4,3))
    k=[1,5,10,20,50,100]
    
    plt.plot(k,img2txt_precision,label='image-to-text',color='blue')
    plt.plot(k,txt2img_precision,label='text-to-image',color='red')

    plt.scatter(k,img2txt_precision,color='blue',s=50)
    plt.scatter(k,txt2img_precision,color='red',s=50)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.xticks(k, fontsize=8)
    plt.yticks(fontsize=8)

    plt.legend()
    plt.tight_layout()

    plt.savefig('./precision_decline.pdf')

if __name__=='__main__':
    precision1=[1.0,1.0,.5,.25,.1,.05]
    precision2=[1.0,0.2,0.1,.05,.02,.01]
    plot_precision_decline(precision1,precision2)


