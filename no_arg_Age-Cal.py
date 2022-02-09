import cv2
import argparse

#-----------Model File Paths----------------#
faceProto="Models/opencv_face_detector.pbtxt"
faceModel="Models/opencv_face_detector_uint8.pb"
ageProto="Models/age_deploy.prototxt"
ageModel="Models/age_net.caffemodel"
genderProto="Models/gender_deploy.prototxt"
genderModel="Models/gender_net.caffemodel"


#-----------Model Variables---------------#
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

#-------------Creating the DNN------------#
faceNet= cv2.dnn.readNet(faceModel,faceProto)
ageNet= cv2.dnn.readNet(ageModel,ageProto)
genderNet= cv2.dnn.readNet(genderModel,genderProto)



#---------Instantiate the Video Capture Object-----------#
video=cv2.VideoCapture(0) #check whether image was passed or not otherwise use the webcam

while cv2.waitKey(1)<0:

    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    #----------------Face Detection-----------------#
    img=frame.copy()
    #--------saving the image dimensions as height and width-------#
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]

    #-----------blob-> Preprocessing the image to required input of the model---------#
    blob=cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob) #setting the image blob as input
    detections = faceNet.forward()
    

    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.7:
            # TopLeftX,TopLeftY, BottomRightX, BottomRightY = inference_results[0, 0, i, 3:7] --> gives co-ordinates bounding boxes for resized small image
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
            # faceBoxes.append(box.astype("int"))
            faceBoxes.append([x1,y1,x2,y2])

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)

    if not faceBoxes:
        print('No face detected')
        break

    for faceBox in faceBoxes:
        #-------Crop out the face from the image---------#
        face=frame[faceBox[1]:faceBox[3],faceBox[0]:faceBox[2]] #img[y1:y2 , x1:x2]

        #------Gender prediction---------#
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
        #-------Age Prediction---------#
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(img, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting age and gender", img)


video.release()
cv2.destroyAllWindows()