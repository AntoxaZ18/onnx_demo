from queue import Queue
from threading import Lock


class ResampleQueue:
    '''
    очередь от видеопотока, делает равномерную выборку если производительности нейросетки не хватает
    '''
    def __init__(self, get_frames=8):
        '''
        get_frames - размер батча для нейронки
        '''
        self.batch_size = get_frames
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

    def get_batch(self):

        with self.lock:
            input_batch = list(self.queue.queue)
            self.queue.queue.clear()

        if len(input_batch) > self.batch_size:    #количество кадров пришло больше чем максимальный размер батча делаем ресемплинг
            step = len(input_batch) / self.batch_size
            resampled = []
            for i in range(self.batch_size):
                resampled.append(input_batch[int(i * step)])
            
            return resampled
        
        return input_batch  #иначе возвращем как есть
