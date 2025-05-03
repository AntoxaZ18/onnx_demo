from queue import Queue
from threading import Lock


class ResampleQueue:
    '''
    очередь от видеопотока, делает равномерную выборку если производительности нейросетки не хватает
    '''
    def __init__(self):
        self.queue = Queue()
        self.lock = Lock()

    def qsize(self):
        return self.queue.qsize()

    def append(self, item):
        self.queue.put(item)

    def get(self):
        if self.queue.qsize():
            return self.queue.get_nowait()
        return None

    def batch(self, batch_size=8):
        """
        return a batch of images
        """
        with self.lock:
            input_batch = list(self.queue.queue)
            self.queue.queue.clear()

        if len(input_batch) > batch_size:    #количество кадров пришло больше чем максимальный размер батча делаем ресемплинг
            step = len(input_batch) / batch_size
            resampled = []
            for i in range(batch_size):
                resampled.append(input_batch[int(i * step)])
            
            return resampled
        
        return input_batch  #иначе возвращем как есть
