We have 2 versions of json file. MMS_hyper.json is use for image input resolution 512x512 which get the best result.
To use MMS_Hyper_Re128 you need to change 
workflow_path="MMS_hyper.json"
to 
workflow_path="MMS_Hyper_Re128.json" in main.py file
MMS_Hyper_Re128 is use for image input resolution 128x128.Result may be hallucinate because x8 upscaling workflow (x2 model x4 model). May luck be with you.