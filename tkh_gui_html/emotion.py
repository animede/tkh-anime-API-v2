def emotion(current_pose_dic,emotions,intensity):

    if emotions=="happy":#喜
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="happy_wink"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="iii"
        current_pose_dic["mouth"]["val"]=intensity         
    elif emotions=="angry":#怒
        current_pose_dic["eyebrow"]["menue"]="angry"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="raised_lower_eyelid"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="uuu"
        current_pose_dic["mouth"]["val"]=intensity     
    elif emotions=="sorrow":#哀
        current_pose_dic["eyebrow"]["menue"]="troubled"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="unimpressed"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="ooo"
        current_pose_dic["mouth"]["val"]=intensity 
    elif emotions=="relaxed":#楽
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="relaxed"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="iii"
        current_pose_dic["mouth"]["val"]=1-intensity             
    elif emotions=="smile":#微笑む
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="relaxed"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="aaa"
        current_pose_dic["mouth"]["val"]=intensity              
    elif emotions=="laugh":#笑う
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="wink"
        current_pose_dic["eye"]["left"]=1-intensity
        current_pose_dic["eye"]["right"]=1-intensity
        current_pose_dic["mouth"]["menue"]="aaa"
        current_pose_dic["mouth"]["val"]=intensity            
    elif emotions=="surprised":#驚く
        current_pose_dic["eyebrow"]["menue"]="lowered"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="surprised"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="ooo"
        current_pose_dic["mouth"]["val"]=intensity              
    else:
        print("Emotion Error")
    print(current_pose_dic)
    return current_pose_dic
