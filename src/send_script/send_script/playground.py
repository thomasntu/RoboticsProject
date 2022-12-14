import argparse

import cv2


def main():
    args = parse_args()

    img = cv2.imread(args.input)
    assert img is not None

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 1, 1)

    img = cv2.Canny(img, 10, 100)

    cv2.imshow("path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """
    TODO
    """
    parser = argparse.ArgumentParser(description="Document detector")
    parser.add_argument('-i', '--input', help="Path to image")

    return parser.parse_args()


if __name__ == "__main__":
    main()
