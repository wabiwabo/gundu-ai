import pika
import os
import dotenv

dotenv.load_dotenv(override=True)

def rabbitmq_connect():
    credentials = pika.PlainCredentials(os.getenv('RABBITMQ_USERNAME'), os.getenv('RABBITMQ_PASSWORD'))
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(os.getenv('RABBITMQ_IP'), int(os.getenv('RABBITMQ_PORT')), '/', credentials))
    channel = connection.channel()
    return connection, channel

def rabbitmq_declare_queue(queue):
    connection, channel = rabbitmq_connect()
    channel.queue_declare(queue=queue)
    connection.close()

def rabbitmq_publish(message, queue):
    connection, channel = rabbitmq_connect()
    channel.basic_publish(exchange='', routing_key=queue, body=message)
    connection.close()