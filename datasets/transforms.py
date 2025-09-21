from torchvision import transforms as T
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt
from torchvision import transforms as T




def pair(x):
    return x, x


def transform(resolution=256, is_train=True):
    ops = []
    # Resize shortest side to resolution, then center-crop to exact square
    # (many repos just CenterCrop if images were pre-resized; this is robust)
    ops += [T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution)]
    if is_train:
        ops += [T.RandomHorizontalFlip(p=0.5)]  # optional but common
    ops += [
        T.ToTensor(),                      # [0,1]
        T.Normalize(mean=(0.5,0.5,0.5),    # -> [-1,1]
                    std=(0.5,0.5,0.5)),
    ]
    return T.Compose(ops)




if __name__=="__main__":
    import cv2
    import numpy as np
    from omegaconf import DictConfig, ListConfig, OmegaConf

    while True:
        transform_train = transform(resolution=256, train=True)
        img = Image.open("/home/pranoy/code/TexTok/imagaes/24394227.jpeg").convert("RGB")
        img_transformed = transform_train(img)

        # convert to numpy for visualization
        img = transforms.ToPILImage()(img_transformed)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)
        cv2.waitKey(0)

        print(img_transformed.shape)


        # img_numpy = img_transformed.view(-1).numpy()
        # show the distribution of the image
        # plt.hist(img_numpy, bins=100)
        # # save the image
        # plt.savefig("hist.jpg")
        

        # grid = make_grid(img_transformed, nrow=6, normalize=True, value_range=(-1, 1))
        # # save_image(grid, "t.jpg")
        # grid = grid.permute(1, 2, 0).detach().cpu().numpy()
        # cv2.imshow("img", grid)
        # # cv2.imwrite("t.jpg", grid)
        # # break
        
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break