from model.enc_dec import setup_codec
from utils.data import ImagesDataloader, human_order, tomat, topil, cv2pil
from torch.utils.data import DataLoader
import cv2
from pathlib import Path
from PIL import Image

save_format = "img%03d"

def main():
    
    in_path = '/home/markm536/Huawei/mbtsr/imgs/test'
    out_path = Path('/home/markm536/Huawei/mbtsr/results')

    model = setup_codec()
    dataloader = ImagesDataloader(in_path)
    train_dataloader = DataLoader(dataloader, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)

    for i, img in enumerate(train_dataloader):
        enc_res = model.compress(img)
        dec_res = model.decompress(**enc_res)['x_hat'].squeeze(0).detach()

        # img = img.squeeze(0)
        img, dec_res = human_order(img), human_order(dec_res)

        im_gt   = topil(img)
        im_dist = topil(dec_res)
        # print(im_gt.max(), im_gt.mean(), im_gt.min(), im_dist.max(), im_dist.mean(), im_dist.min())
        # print(im_gt.shape, im_dist.shape)

        im_gt.save(out_path / f"{save_format % i}_gt.png")
        im_dist.save(out_path / f"{save_format % i}_dist.png")

        # sbs = cv2.hconcat([im_dist, im_gt])
        # cv2.imwrite(str(out_path / f"{save_format % i}_gt.png"), im_gt) 
        # cv2.imwrite(str(out_path / f"{save_format % i}_dist.png"), im_dist) 



if __name__ == "__main__":
    main()