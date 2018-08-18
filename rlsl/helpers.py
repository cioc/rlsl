import boto3

class CloudWatchMetrics(object):
    def __init__(self, namespace):
        self.namespace = namespace
        self.cw = boto3.client('cloudwatch',
                               region_name='us-west-2')

    def write(self, metrics):
        self.cw.put_metric_data(Namespace=self.namespace,
                                MetricData=metrics)

class LocalMetrics(object):
    def __init__(self):
        pass

    def write(self, metrics):
        for m in metrics:
            print(m)

class S3FS(object):
    def __init__(self, bucket):
        self.bucket = bucket
        self.s3 = boto3.client('s3',
                               region_name='us-west-2')

    def write(self, key, data):
        self.s3.put_object(Bucket=self.bucket,
                           Key=key,
                           Body=data)

def LocalFS(object):
    def __init__(self, bucket):
        pass

    def write(self, key, data):
        with open(key, "w+") as f:
            f.write(data)
