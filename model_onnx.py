from time import time
import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from tracker import TrackerInput, ByteTracker

def get_providers():
    """
    return available providers (only cpu and cuda supplied)
    """
    providers = [i for i in ort.get_available_providers() if any(val in i for val in ('CUDA', 'CPU'))]
    modes = {
        'CUDAExecutionProvider': 'gpu',
        'CPUExecutionProvider': 'cpu'
    }
    providers = [modes.get(i) for i in providers]
    return providers


class YoloONNX():
    def __init__(self, path: str, session_options=None, device='cpu', batch=1, confidence=0.5, labels=None) -> None:

        if not labels:
            raise ValueError('Provide non empty list in "labels" parameter')

        sess_options = ort.SessionOptions()
        #optimize onnx provider parameters
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode  = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.inter_op_num_threads = 3

        sess_providers = ['CPUExecutionProvider']
        if device == 'gpu':
            sess_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.mode = 'gpu'
        else:
            self.mode = 'cpu'
        
        self.session = ort.InferenceSession(path, providers=sess_providers, sess_options=sess_options)    #'CUDAExecutionProvider',

        self.input_width = 640
        self.input_height = 640
        self.batch = batch

        self.iou = 0.8
        self.confidence_thres = confidence
        self.classes = labels
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.executor = ThreadPoolExecutor(max_workers=self.batch)

        self.tracker = ByteTracker()

    def _image_batch_preprocess(self, image_batch: List):
        images = [self.letterbox(img) for img in image_batch]
        imgs = [img[0] for img in images]
        pads = [img[1] for img in images]
        
        imgs = np.stack(imgs)
        imgs = np.ascontiguousarray(imgs)  # contiguous array for better performance

        imgs = imgs[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        imgs = imgs.astype(np.float32) / 255.0
        return imgs, pads

    def _image_preprocess(self, rgb_frame) -> np.ndarray:
        """image preprocessing
        rgb_frame - image in rgb format
        including resizing to yolo input shape
        add batch dimension and normalized to 0...1 range
        convert from image to tensor view
        # """
        self.img_height, self.img_width = rgb_frame.shape[:2]

        # image_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        image_data, pad = self.letterbox(rgb_frame, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(image_data) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data, pad
    
    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int, track_id: int=0) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (List[float]): Detected bounding box coordinates [x, y, width, height].
            score (float): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        # Extract the coordinates of the bounding box
        box = [int(i) for i in box]
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f} id:{track_id}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    def postprocess(self, output: List[np.ndarray], pad: Tuple[int, int]) -> np.ndarray:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.
        """
        # Transpose and squeeze the output to match the expected shape

        outputs = np.transpose(np.squeeze(output[0]))

        gain = np.float32(min(self.input_height / self.img_height, self.input_width / self.img_width))
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        #find max score for prediction
        max_scores = np.amax(outputs[:, 4:], axis=1)
        #filter prediction with result more than confidence thresh
        results = outputs[max_scores > self.confidence_thres]
        #find class ids
        class_ids = np.argmax(results[:, 4:], axis=1).tolist()
        scores = np.amax(results[:, 4:], axis=1).tolist()
        
        #convert coordinates of boxes
        x = results[:, 0]
        y = results[:, 1]
        w = results[:, 2]
        h = results[:, 3]

        left = ((x - w / 2) / gain).astype(int)
        top = ((y - h / 2) / gain).astype(int)
        width = (w / gain).astype(int)
        height = (h / gain).astype(int)

        boxes = np.column_stack((left, top, width, height)).tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou)

        predictions = []

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            predictions.append([*box, score, class_id])

        return predictions



    def __call__(self, image_batch: np.ndarray) -> np.ndarray:
        """return image of object if they are on image. Return only one object with highest score"""
        if self.mode == 'gpu':
            res = self.call_gpu(image_batch)
            return res
        else:
            return self.call_cpu(image_batch)
    
    def call_gpu(self, image_batch:List[np.ndarray]):
        '''call gpu variant'''

        futures = [self.executor.submit(self._image_preprocess, image) for image in image_batch]

        wait(futures, return_when=ALL_COMPLETED)
        tensor_images = [f.result() for f in futures]

        output_name = self.session.get_outputs()[0].name
        input_name = self.session.get_inputs()[0].name

        imgs = [img for (img, _) in tensor_images]
        pads = [pad for (_, pad) in tensor_images]

        batch = np.concatenate(imgs, axis=0)
        
        outputs = self.session.run([output_name], {input_name: batch})
        
        predictions = outputs[0]

        results = [self.postprocess(np.expand_dims(predictions[idx], axis=0), pad) for image, idx, pad in zip(image_batch, iter(range(predictions.shape[0])), pads)]

        return self.draw(image_batch, results)


    def process(self, image: np.ndarray):
        tensor, pad = self._image_preprocess(image)
        output_name = self.session.get_outputs()[0].name
        input_name = self.session.get_inputs()[0].name

        outputs = self.session.run([output_name], {input_name: tensor})
        return self.postprocess(outputs, pad)



    def call_cpu(self, image_batch:List[np.ndarray]):
        '''
        call cpu variant on image batch
        '''

        futures = [self.executor.submit(self.process, image) for image in image_batch]
        wait(futures, return_when=ALL_COMPLETED)
        predictions = [f.result() for f in futures]

        return self.draw(image_batch, predictions)



    def draw(self, image_batch:List[np.ndarray], predictions:List):
        '''draw boxes on images'''
        
        for img, img_preds in zip(image_batch, predictions):

            if not img_preds:
                continue

            dets = np.array(img_preds)

            x = TrackerInput(conf=dets[:, 4], xywh=dets[:, 0:4], cls_=dets[:, 5])

            online_targets = self.tracker.update(x, (self.img_height, self.img_width))

            if len(online_targets):
                x1 = online_targets[:, 0]
                y1 = online_targets[:, 1]
                x2 = online_targets[:, 2]
                y2 = online_targets[:, 3]

                targets = np.column_stack([(x1+x2) / 2, (y1 + y2) / 2, x2-x1, y2-y1, online_targets[:, 5], online_targets[:, 6], online_targets[:, 4]]).tolist()

                for box in targets:
                    self.draw_detections(img, [box[0], box[1], box[2], box[3]], box[4], int(box[5]), int(box[6]))

        return image_batch
    


if __name__ == '__main__':

    print(ort.get_available_providers())

    batch_images = 12

    nano = 'yolo11n_5epoch_16batch640.onnx'
    small = 'y11_100ep16b640.onnx'

    model = YoloONNX(small, device='cpu', batch = batch_images)

    frame = cv2.imread('test.jpg')
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    images = [image_rgb] * batch_images
    process_imgs = model(images)

    print(ort.get_available_providers())

    cv2.startWindowThread()

    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('frame', process_imgs[0])
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # cv2.imwrite('out.jpg', process_imgs[0])
    # cv2.imshow('0', process_imgs[0])

    # sys.exit(0)
