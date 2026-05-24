import os, torch, numpy as np
from PIL import Image
from torchvision import transforms
import timm, torch.nn as nn

backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=64)
feat_dim = backbone.embed_dim
model = nn.Sequential(backbone, nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2)).cuda()
model.load_state_dict(torch.load('best_dinov2_eye.pth'))
model.eval()

tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

for cls_name, label in [('awake',0),('sleepy',1)]:
    d = f'/home/bobby/eye_datasets/data/test/{cls_name}'
    fnames = sorted(os.listdir(d))[:2000]
    confs, wrong = [], []
    for fname in fnames:
        img = Image.open(os.path.join(d, fname)).convert('L').convert('RGB')
        x = tf(img).unsqueeze(0).cuda()
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)
            conf = p.max().item()
            pred = p.argmax().item()
            confs.append(conf)
            if pred != label:
                wrong.append((conf, pred, fname))
    ca = np.array(confs)
    correct = len(confs) - len(wrong)
    print(f'{cls_name}: n={len(confs)} acc={correct/len(confs)*100:.1f}% mean_conf={ca.mean():.4f} min={ca.min():.4f} med={np.median(ca):.4f}')
    print(f'  <0.95:{sum(c<0.95 for c in ca)} <0.9:{sum(c<0.9 for c in ca)} <0.8:{sum(c<0.8 for c in ca)} <0.7:{sum(c<0.7 for c in ca)} <0.6:{sum(c<0.6 for c in ca)}')
    if wrong:
        wrong.sort()
        print(f'  Misclassified ({len(wrong)}):')
        for c,p,f in wrong[:5]:
            print(f'    {f} conf={c:.4f} pred={p}')
print('DONE')
