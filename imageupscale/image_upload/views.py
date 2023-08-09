from django.shortcuts import render
from PIL import Image
from django.http import HttpResponse
import torch
from PIL import Image
from .RealESRGAN import RealESRGAN
def imageUpload(request):
    flag = True
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        scale = request.POST.get('Scale')
        image = uploaded_image
        print(scale)
        pil_image = Image.open(image)
        pil_image = pil_image.convert('RGB')
        print(pil_image)
        if torch.cuda.is_available():   
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = RealESRGAN(device, scale=int(scale[0]))
        path = 'weights/RealESRGAN_x' + str(scale[0]) + '.pth'
        model.load_weights(path, download=True)
        print('before predict')
        SuperResolution_image = model.predict(pil_image)
        print('after Predict') 
        response = HttpResponse(content_type='image/jpeg')
        SuperResolution_image.save(response, 'jpeg')
        return response
    return render(request, 'image_upload/index.html', {'flag': flag})
    
