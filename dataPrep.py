import cv2
import inspect
import os
import csv
import numpy as np
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
# from Basilisk import __path__
# bskPath = __path__[0]
dirName = os.path.abspath(os.path.dirname(__file__)) + "\CoB_data"

def grayscale(image_path: str) -> None:
    # Read the image from the given path
    image = cv2.imread(image_path)
    
    # Check if the image was successfully read
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Return the path of the grayscale image
    return grayscale_image


def CoB(image):
    # Read the image
    
    # Calculate the moments of the image
    moments = cv2.moments(image)
    
    # Calculate the x and y coordinates of the centroid
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = 0, 0
    
    return cx, cy

def imgConv(image):
    # Read the grayscale image    
    # Check if the image was successfully read
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    # Flatten the image to a 1D vector
    image_vector = image.flatten()
    # print(image_vector.shape)
    
    return image_vector

if __name__ == "__main__":
    output_dir=os.path.join(os.getcwd(), 'Mars\\data')
    os.makedirs(output_dir, exist_ok=True)

    
    # Iterate through all files in the directory
    for i in range(509):
        directory= os.path.join(os.getcwd(), 'cnn_MC_data\\run'+str(i)) 
        csvfile = open(output_dir + "\\run" + str(i) + "CoB.csv", 'w') 
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'X_c', 'Y_c'])

        rows=[]
        Cx=[]
        Cy=[]
        for filename in os.listdir(directory):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Check if the file is an image (you can add more extensions if needed)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # Convert to grayscale
                    img = grayscale(file_path)
                    # Find the center of brightness
                    x, y = CoB(img)
                    # print(f"Center of brightness for {filename}: ({x}, {y})")
                    Cx.append(x)
                    Cy.append(y)
                    # img_p=imgConv(img)
                    # array=np.append(img_p,np.array([x,y]))
                    # array=array.tolist()
                    # array.insert(0,filename)
                    rows.append([filename, x, y])
                    

                except Exception as e:
                    print(f"Error processing {filename}: {e}")    
        # print('done: '+str(i))        
        rows.sort(key=lambda row: float(row[0].split('.')[0]))
    
        # Modification 3: Write sorted rows to CSV
        writer.writerows(rows)
        
        csvfile.close()

    
    