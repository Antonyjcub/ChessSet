import torch
import cv2
import math
import numpy
import chess
import requests
import glob
import os
import pandas as pd
from typing import Dict
SQUARE_WIDTH = 0.125
ASCII_OFFSET = 96

LEGEND = ['BLACK KNIGHT', 'BLACK BISHOP', 'BLACK KING', 'BLACK PAWN', 'BLACK QUEEN', 'BLACK ROOK', 'WHITE BISHOP', 'WHITE KING', 'WHITE PAWN', 'WHITE QUEEN', 'WHITE ROOK', 'WHITE KNIGHT']
PIECE_MAP = {
    "PAWN": chess.PAWN,
    "KNIGHT": chess.KNIGHT,
    "BISHOP": chess.BISHOP,
    "ROOK": chess.ROOK,
    "QUEEN": chess.QUEEN,
    "KING": chess.KING
}
REV_PIECE_MAP = {
    "p": "PAWN",
    "n": "KNIGHT",
    "b": "BISHOP",
    "r": "ROOK",
    "q": "QUEEN",
    "k": "KING"
}

COLOR_MAP = {
    "BLACK": chess.BLACK,
    "WHITE": chess.WHITE,
}
label ='labels.txt'
def chessDetect(
    image_path: str,
    model_path: str,
    repo_path: str,
    label_output_path: str,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5
):
    """
    Runs YOLOv5 object detection on a chessboard image.
    Returns:
        results (torch.Tensor or pandas.DataFrame): Model detection results.
        image (numpy.ndarray): The original input image as loaded by OpenCV.
    """

    # Load YOLOv5 model
    model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

    # Set inference settings
    model.conf = conf_thresh
    model.iou = iou_thresh
    model.agnostic = True

    # Read image and run inference
    image = cv2.imread(image_path)
    results = model(image_path)

    # Optional: save label output or do something with results
    results.save()  # saves rendered images to 'runs/detect/exp'

    return results, image

def get_latest_label_file() -> str:
    label_dirs = sorted(glob.glob('runs/detect/exp*/labels'), key=os.path.getmtime, reverse=True)
    if not label_dirs:
        raise FileNotFoundError("No label directories found.")
    
    label_files = glob.glob(os.path.join(label_dirs[0], '*.txt'))
    if not label_files:
        raise FileNotFoundError("No label files found in latest directory.")

    return label_files[0]

def save_labels(results, label_path: str):
    df = results.pandas().xyxy[0]
    df.to_csv(label_path, index=False, header=True)

def sortNLocate(label_csv_path, width, height) -> Dict[int, chess.Piece]:
    df = pd.read_csv(label_csv_path)
    PieceNPlace = {}

    for _, row in df.iterrows():
        xMin, yMin, xMax, yMax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        center_x = (xMin + xMax) / 2
        center_y = (yMin + yMax) / 2

        file = math.ceil(center_x / (width * SQUARE_WIDTH))
        file = chr(ASCII_OFFSET + file)

        rank = 9 - math.ceil(center_y / (height * SQUARE_WIDTH))

        try:
            position = chess.parse_square(f"{file}{rank}")
        except:
            print(f"Skipping invalid square: {file}{rank}")
            continue

        chessClass = int(row['class'])
        pieceSplit = LEGEND[chessClass].split()
        color = COLOR_MAP[pieceSplit[0].upper()]
        piece = PIECE_MAP[pieceSplit[1].upper()]
        piece_obj = chess.Piece(piece, color)

        PieceNPlace[position] = piece_obj

    return PieceNPlace

def flip_pos(pos):
    file, rank = pos[0], pos[1]
    if not ('a' <= file <= 'h') or not ('1' <= rank <= '8'):
        raise ValueError("Invalid file or rank")
    flipped_file = chr(ord('h') - (ord(file) - ord('a')))
    flipped_rank = chr(ord('1') + ord('8') - ord(rank))
    return flipped_file + flipped_rank



results, im = chessDetect(
    image_path="/Users/antonyjacob/Desktop/ChessSet/chess.png",
    model_path="/Users/antonyjacob/Desktop/ChessSet/yolov5/runs/train/chess_detect4/weights/best.pt",
    repo_path="/Users/antonyjacob/Desktop/ChessSet/yolov5",
    label_output_path="label.txt"
)


# Display results
#results.print()
results.show()
results.save()
height, width = im.shape[:2]

height, width = im.shape[:2]
save_labels(results, "labels.txt")
df = pd.read_csv("labels.txt")
PieceNPlace = sortNLocate("labels.txt", width, height)

board = chess.Board.empty()
board.set_piece_map(PieceNPlace)
fen = board.board_fen()
#Get input for whose turn for now hardcoded 
color = input("Whose turn is it: ")
color = color.upper()

color1 = COLOR_MAP[color]

board.turn= color1
board.set_castling_fen("-")
#Sets ep to none 
board.ep_square = None

#sets halfmove and full move clock 
board.halfmove_clock = 0 
board.fullmove_number = 1
if color == "BLACK":
    fen += " b "
else:
    fen += " w "
fen += "- - 0 1"
print(fen)
url = "https://stockfish.online/api/s/v2.php"
payload = {"fen": fen, "depth": 10}
r = requests.get(url, params=payload)
response = r.json()

if not response["success"]:
    print("Bad Request. Try again.")
else:
    bestMove = response["bestmove"].split()[1]
    startPosSTR, endPos = bestMove[:2], bestMove[2:]
    squareInd = chess.parse_square(startPosSTR)
    #print(startPosSTR)
    #print(squareInd)
    pieceToMove = PieceNPlace[squareInd]
    if color == "BLACK":
        startPosSTR = flip_pos(startPosSTR)
        endPos = flip_pos(endPos)

    pieceColor = "White" if pieceToMove.color else "Black"
    specificPiece = chess.piece_name(pieceToMove.piece_type)
    
    print(f"Move your {pieceColor} {specificPiece} from {startPosSTR} to {endPos}")
