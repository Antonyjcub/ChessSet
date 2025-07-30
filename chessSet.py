"""
chessSet.py.

Runs a chess piece detection pipeline using YOLOv5 to identify piece positions
on a chessboard image. Converts detected positions into a legal FEN string,
queries Stockfish for the best move, and prints move instructions.
"""

import math
import glob
import os
from typing import Dict

import torch
import cv2
import numpy
import chess
import requests
import pandas as pd
SQUARE_WIDTH = 0.125
ASCII_OFFSET = 96

LEGEND = [
            'BLACK KNIGHT', 'BLACK BISHOP', 'BLACK KING',
            'BLACK PAWN', 'BLACK QUEEN', 'BLACK ROOK',
            'WHITE BISHOP', 'WHITE KING', 'WHITE PAWN',
            'WHITE QUEEN', 'WHITE ROOK', 'WHITE KNIGHT'
]
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
label = 'labels.txt'

URL = "https://stockfish.online/api/s/v2.php"


def chessBoardDetect(raw_image_path: str, output_path: str):
    """
    Run YOLOv5 object detection on a window screenshot identifies the chessboard.

    Returns:
        results (torch.Tensor or pandas.DataFrame): Model detection results.
        image (numpy.ndarray): The original input image as loaded by OpenCV.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../ChessSet/yolov5/runs/train/chessboard_id6/weights/best.pt',
     force_reload=True)
    results = model(raw_image_path)
    df = results.pandas().xyxy[0]
    img = cv2.imread(raw_image_path)

    if not df.empty:
        top_box = df.sort_values('confidence', ascending=False).iloc[0]
        xmin, ymin, xmax, ymax = map(int, [top_box['xmin'], top_box['ymin'], top_box['xmax'], top_box['ymax']])
        cropped = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(output_path, cropped)
        print(f"Cropped image saved to: {output_path}")
    else:
        raise ValueError("No chessboard detected.")
    # new content above

def chessDetect(
    image_path: str,
    model_path: str,
    repo_path: str,
    label_output_path: str,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5
):
    """
    Run YOLOv5 object detection on a chessboard image.

    Returns:
        results (torch.Tensor or pandas.DataFrame): Model detection results.
        image (numpy.ndarray): The original input image as loaded by OpenCV.
    """
    # Load YOLOv5 model
    model = torch.hub.load(
        repo_path, 'custom',
        path=model_path, source='local'
    )

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

def save_labels(results, label_path: str):
    """
    Save YOLOv5 detection results as a CSV file.

    Converts the detection results from the YOLOv5 model into a pandas DataFrame
    and saves it to the specified path in CSV format.
    """
    df = results.pandas().xyxy[0]
    df.to_csv(label_path, index=False, header=True)


def sortNLocate(label_csv_path, width, height) -> Dict[int, chess.Piece]:
    """
    Parse detection labels and maps them to chessboard positions.

    Reads bounding box coordinates from a YOLOv5-generated label CSV,
    calculates the center of each box, and converts it to a chess square
    based on board dimensions. Returns a dictionary mapping each square
    to its corresponding chess piece.

    Args:
        label_csv_path (str): Path to the label CSV file.
        width (int): Width of the input image in pixels.
        height (int): Height of the input image in pixels.

    Returns:
        Dict[int, chess.Piece]: Mapping from board index (0–63) to chess piece.
    """
    df = pd.read_csv(label_csv_path)
    PieceNPlace = {}

    for _, row in df.iterrows():
        xMin, yMin, xMax, yMax = (
            row['xmin'], row['ymin'], row['xmax'], row['ymax']
            )
        center_x = (xMin + xMax) / 2
        center_y = (yMin + yMax) / 2

        file = math.ceil(center_x / (width * SQUARE_WIDTH))
        file = chr(ASCII_OFFSET + file)

        rank = 9 - math.ceil(center_y / (height * SQUARE_WIDTH))
        position = chess.parse_square(f"{file}{rank}")

        chessClass = int(row['class'])
        pieceSplit = LEGEND[chessClass].split()
        color = COLOR_MAP[pieceSplit[0].upper()]
        piece = PIECE_MAP[pieceSplit[1].upper()]
        piece_obj = chess.Piece(piece, color)

        PieceNPlace[position] = piece_obj

    return PieceNPlace


def flip_pos(pos):
    """
    Flip a square position across both axes of the chessboard.

    Args:
        pos (str): Square in algebraic notation (e.g., 'e2').

    Returns:
        str: Flipped square (e.g., 'd7').
    """
    file, rank = pos[0], pos[1]
    if not 'a' <= file <= 'h' or not '1' <= rank <= '8':
        raise ValueError("Invalid file or rank")
    flipped_file = chr(ord('h') - (ord(file) - ord('a')))
    flipped_rank = chr(ord('1') + ord('8') - ord(rank))
    return flipped_file + flipped_rank

chessBoardDetect(
    raw_image_path="../ChessSet/chess.png",
    output_path="../ChessSet/cropped_chessboard.jpg"
)


results, im = chessDetect(
    image_path="../ChessSet/cropped_chessboard.jpg",
    model_path=(
        "../ChessSet/yolov5/"
        "runs/train/chess_detect4/weights/best.pt"
    ),
    repo_path="../ChessSet/yolov5",
    label_output_path="label.txt"
)
"""Display results
results.print()
"""
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
color = input("Whose turn is it: ")
color = color.upper()

color1 = COLOR_MAP[color]

board.turn = color1
board.set_castling_fen("-")
board.ep_square = None
board.halfmove_clock = 0
board.fullmove_number = 1
if color == "BLACK":
    fen += " b "
else:
    fen += " w "
fen += "- - 0 1"
print(fen)
payload = {"fen": fen, "depth": 10}
r = requests.get(URL, params=payload)
response = r.json()

if not response["success"]:
    print("Bad Request. Try again.")
else:
    bestMove = response["bestmove"].split()[1]
    startPosSTR, endPos = bestMove[:2], bestMove[2:]
    squareInd = chess.parse_square(startPosSTR)
    pieceToMove = PieceNPlace[squareInd]
    if color == "BLACK":
        startPosSTR = flip_pos(startPosSTR)
        endPos = flip_pos(endPos)

    pieceColor = "White" if pieceToMove.color else "Black"
    specificPiece = chess.piece_name(pieceToMove.piece_type)
    print(
        f"Move your {pieceColor} {specificPiece} "
        f"from {startPosSTR} to {endPos}"
    )
