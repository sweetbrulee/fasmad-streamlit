from service.message_queue import MetadataQueueService

class AlarmAgent:
    def __init__(self):
        pass

    def merge(self, alarm_batch: list):
        queue = MetadataQueueService.use_queue()
        queue.get()

    def flush(self):
        # flush metadata queue, which then trigger the send, merge, output pipeline
        # you can call this function after every one frame inference is done, or multiple frame inferences are, it's up to your design.


        pass

    def add_send_rule(self, pred_rule: dict):
        # each pred rule is a callable function which returns a bool
        # will iterate metadata queue and filter out the alarms that match the rule, then send them

        pass

    def del_send_rule(self, keys: list):

        pass

    def add_merge_rule(self, pred_rule: dict):
        # remember the order: send the batch -> merge the batch -> output results on the screen

        pass

    def del_merge_rule(self, keys: list):

        pass
