# YOLO

BETA-UDGAVE!!!!  
Denne dokumentation er ufærdig.

Træning af en model til objektgenkendelse og klassifikation med YOLO og Tensorflow.   

## Annotering
- tag billeder af terninger
- upload til makesense.ai
- annoter med makesense.ai
- download annotations
- konverter til YOLO format (mappestruktur)

```
datasets
    images
        train
        val
    labels
        train
        val
```

## Træning af objektdetektering
Script: 1-train_model.py

- træn objektdetektering med yolo
- test model på billeder  

**Output:** `best.pt` i `runs/detect/trainX/weights`

## Cropping
Script: 2-crop_images.py  
- crop billedfiler efter bounding boxes

**Output:** Billeder i filstruktur

Bruger første bogstav i filnavn til undermapper. Omskriv kode som nødvendigt.
```   
   # Save the output image
    output_path_cropped = os.path.join("output", f"cropped_{image_name}")
    folder = image_name[0]
    output_path_cropped = os.path.join("output", folder, f"cropped_{image_name}")
 ```

## Træning af klassifikation
Script: 3-train_classifier.py  

Brug modellen fra før som input til træningen. 

- træn model på croppede billeder.



**Output:** best.pt  i `runs/detect/trainX/weights`

## Afprøvning i browser
Brug Tensorflow script, `tm.html`.
Man er måske nødsaget til at konvertere ens model til tensorflow.

- læg terning under kamera
- med model 1 genkend objekt
- med model 2 genkend terning 

