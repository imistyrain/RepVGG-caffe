import cv2
import numpy as np

model_dir = "models"
model_name = "RepVGG-A0"

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

def preprocess(img):
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img)/255.0
    img[:,:,]-=[0.485, 0.456, 0.406]
    img[:,:,]/=[0.229, 0.224, 0.225]
    return img

def print_caffe_featrues():
    with open("output/caffe_outputs.txt","w") as f:
        for feature in net.blobs:
            data = net.blobs[feature].data.squeeze()#.reshape(-1,4)
            f.write(feature+str(data.shape)+"\n")
            f.write(str(data)+"\n")

def demo_caffe(img):
    print("demo caffe")
    import caffe
    np.set_printoptions(precision=6, suppress=True)
    net = caffe.Net(model_dir+"/"+model_name+".prototxt",model_dir+"/"+model_name+".caffemodel",caffe.TEST)
    net.blobs[net.inputs[0]].data[...] = preprocess(img).transpose(2,0,1)
    net.forward()
    pred = net.blobs[net.outputs[0]].data[0]
    index = np.argsort(pred)[::-1]
    for i in range(5):
        print(index[i], classes[index[i]], pred[index[i]])

def demo_dnn(img):
    print("demo dnn")
    net = cv2.dnn.readNet(model_dir+"/"+model_name+".prototxt",model_dir+"/"+model_name+".caffemodel")
    img = preprocess(img)
    blob = cv2.dnn.blobFromImage(img)
    net.setInput(blob)
    pred = net.forward()[0]
    index = np.argsort(pred)[::-1]
    for i in range(5):
        print(index[i], classes[index[i]], pred[index[i]])

def print_weights_all(model):
    np.set_printoptions(precision=6, suppress=True)
    with open("output/pytorch_weights.txt", "w") as f:
        parameters = model.state_dict()
        for name in parameters:
            parameter = parameters[name].detach().cpu().numpy().squeeze()
            shape = parameter.shape
            print(name, shape)
            f.write(str(name)+str(shape)+"\n")
            print(parameter)
            f.write(str(parameter)+"\n")

def get_features(model, img):
    import torch
    np.set_printoptions(precision=6, suppress=True)
    features = {}
    hooks = []
    layer_instances = {}
    def add_hooks(module):
        def hook(module, input, output):
            instance_index = 1
            class_name = str(module.__class__.__name__)
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index
            layer_name = class_name + "_"+str(instance_index)
            features[layer_name] = output
        if not isinstance(module, torch.nn.ModuleList) and not isinstance(module, torch.nn.Sequential) and module != model:
            hooks.append(module.register_forward_hook(hook))
    model.apply(add_hooks)
    model(img)
    with open("output/features.txt","w") as f:
        input_np = img.detach().cpu().numpy().squeeze()
        f.write("blob1 "+str(input_np.shape)+"\n")
        f.write(str(input_np)+"\n")
        for feature in features:
            fvalues = features[feature].detach().cpu().numpy().squeeze()
            print(feature, fvalues.shape)
            print(fvalues)
            f.write(feature+" "+str(fvalues.shape)+"\n")
            f.write(str(fvalues)+"\n")

def demo_pytorch(img):
    print("demo pytorch")
    import torch
    from repvgg import repvgg_model_convert, create_RepVGG_A0
    model = create_RepVGG_A0(deploy=True)
    model.load_state_dict(torch.load(model_dir+'/'+model_name+'-deploy.pth'))
    
    img = preprocess(img)
    img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    #print_weights_all(model)
    #get_features(model, img)
    pred = model(img)
    _, indices = torch.sort(pred, descending=True)
    percentage = torch.nn.functional.softmax(pred, dim=1)[0]
    [print(idx.detach().cpu().item(), classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

if __name__=="__main__":
    img = cv2.imread("images/cat.jpg")
    demo_caffe(img)
    demo_dnn(img)
    demo_pytorch(img)