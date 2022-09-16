from telnetlib import PRAGMA_HEARTBEAT
import face_recognition 
import cv2 
import numpy as np  
import os 

class Data: 

    def __init__(self , face_image_path) -> None:
        self.face_image = face_recognition.load_image_file(face_image_path) 
        self.face_encoding = face_recognition.face_encodings(self.face_image)[0]  
        self.display_name = "" 
        for letter in os.path.basename(face_image_path) : 
            if letter != ".": 
                self.display_name = self.display_name + letter 
            else :
                break  


        self.imshow = [self.display_name , (self.face_image , self.face_encoding)]   

    def __repr__(self) -> str:
        return self.display_name
    
    def __str__(self):  
        return self.__repr__()



class Face_Recognition_WebCam: 

    def __init__(self) -> None:
        self.initialise_data() 
        self.frameWidth , self.frameHeight = 600 , 800 
        
        

    def initialise_data(self , dataPath = "../img/"): 
        self.dataset = []   
        self.knownEncodings = [] 
        if dataPath[-1] != "/" or dataPath[-1] != "\\" : dataPath  += "/" 

        
        for img_name in os.listdir(dataPath):  
            data = Data(dataPath + img_name)
            self.dataset.append(data) 
            self.knownEncodings.append(data.face_encoding)  
    
    def start_Recognition(self): 
        
        running = True   
        process_this_frame = True 
        capture = cv2.VideoCapture(0)
        while running  and capture.isOpened(): 
            success , cv2_bgr_frame = capture.read()   

            # To make the regocnition faster , make the size smaller by percentage , dont do (width,height) 
            small_frame = cv2.resize(cv2_bgr_frame , (0,0) , fx=0.25 , fy=0.25 ) 

            #rgb_frame = cv2.cvtColor(small_frame , cv2.COLOR_BGR2RGB)  
            rgb_frame = small_frame[:,:, ::-1] # face_recognition function uses rgb images , thats why we are converting  

            if process_this_frame : 

                face_locations = face_recognition.face_locations(rgb_frame) 
                face_encodings = face_recognition.face_encodings(rgb_frame , face_locations) 

                face_names = [] 
                for face_encoding in face_encodings: 

                    matches = face_recognition.compare_faces(self.knownEncodings , face_encoding) 
                    name = "Unknown" 
                    face_distances = face_recognition.face_distance(self.knownEncodings , face_encoding) 
                    best_match_index = np.argmin(face_distances) 
                    if matches[best_match_index]: 
                        name = self.dataset[best_match_index].imshow[0]

                    face_names.append(name) 

            process_this_frame = not process_this_frame 

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(cv2_bgr_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(cv2_bgr_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(cv2_bgr_frame, name, (left + 4, bottom - 4), font, 1, (0,0,255), 2) 

            cv2.imshow("Face Recognition" , cv2_bgr_frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False 

        capture.release()
        cv2.destroyAllWindows()  


if __name__ == "__main__": 
    recognizer = Face_Recognition_WebCam() 
    recognizer.start_Recognition() 







            


    





