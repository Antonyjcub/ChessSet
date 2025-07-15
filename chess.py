import torch
import cv2
import math



model = torch.hub.load(
    '/Users/antonyjacob/Desktop/ChessSet/yolov5', 
    'custom', 
    path='/Users/antonyjacob/Desktop/ChessSet/yolov5/runs/train/chess_detect4/weights/best.pt', 
    source='local')

# Run inference
img = "/Users/antonyjacob/Desktop/ChessSet/chess.png"
results = model(img)
im = cv2.imread(img)

# Display results
results.print()
results.show()
results.save()
print(im.shape)
width = im.shape[0]
height = im.shape[1]

changeToStr = results.pandas().xyxy[0]  
changeToStr = str(changeToStr)
PieceNPlace = []
Legend = ['Black Knight', 'Black Bishop', 'Black King', 'Black Pawn', 'Black Queen', 'Black Rook', 'White bishop', 'White king', 'White pawn', 'White queen', 'White rook', 'White Knight']
with open('labels.txt', 'w') as output:
    output.write(changeToStr)

with open('labels.txt', encoding="utf-8") as f:
    read_data = f.readline()
    print(read_data)
    for line in f:
        line = line.strip()
        if not line:
            break  
        check = line[0]
        line = line.split()
        xMin, yMin, xMax, yMax = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        xValue =  (xMin + xMax)/2
        xValue = (xValue/width)
        xValue = (xValue/.125)
        xValue = math.ceil(xValue)
        xValue = chr(96 + xValue)
        yValue =  (yMin + yMax)/2
        yValue = (yValue/height)
        yValue = (yValue/.125)
        yValue = math.ceil(yValue)
        yValue = 9 - yValue
        chessClass = int(line[6])
        x = (Legend[chessClass],xValue, yValue)
        PieceNPlace.append(x)
print(PieceNPlace)
# Get predictions
#print(results.xyxy[0])  # tensor with [x1, y1, x2, y2, conf, class]
# prediction in class_id center_x center_y width height
# origin is on the Top left side of the img 1 on the right for X
# 1 is the bottom for the 
#names: ['Black Knight', 'black bishop', 'black king', 'black pawn', 'black queen', 'black rook', 'white bishop', 'white king', 'white pawn', 'white queen', 'white rook', 'whiteknight']
#            0              1               2           3               4               5           6               7               8           9                   10              11           
# so a function that reads in the entire line and takes the 1st part as classification 
# the second piece as where on the x axis the the pieces is so divide by .125 and round up to a number
# 