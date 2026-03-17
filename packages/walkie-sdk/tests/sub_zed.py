import time

import cv2
import numpy as np
import zenoh  # Standard zenoh library (already installed)
from zenoh_ros2_sdk import ROS2Subscriber, ZenohSession

# Configuration
ROUTER_IP = "127.0.0.1"
ROUTER_PORT = 7447
DOMAIN_ID = 23  # (Assuming you switched to 0 from our previous debug!)
TOPIC = "/zed_head/zed_node/rgb/color/rect/image"
MSG_TYPE = "sensor_msgs/msg/Image"

# --- Stream Monitoring Variables ---
last_msg_time = 0.0
is_receiving = False
TIMEOUT_SECONDS = 3.0


def scan_active_topics(router_ip, router_port, domain_id, scan_time=2.0):
    """Listens to the Zenoh network for a few seconds to discover active ROS 2 topics."""
    print(
        f"\n🔍 Scanning Domain {domain_id} for active ROS 2 topics for {scan_time} seconds..."
    )

    # Setup standard Zenoh connection
    conf = zenoh.Config()
    conf.insert_json5("connect/endpoints", f'["tcp/{router_ip}:{router_port}"]')
    z_session = zenoh.open(conf)

    discovered_keys = set()

    # Callback to catch any passing data
    def sniff_callback(sample):
        discovered_keys.add(str(sample.key_expr))

    # Subscribe to EVERYTHING on this domain ID
    sub = z_session.declare_subscriber(f"{domain_id}/**", sniff_callback)

    # Wait and listen
    time.sleep(scan_time)

    # Cleanup sniffer
    sub.undeclare()
    z_session.close()

    # Parse and print results
    if not discovered_keys:
        print(
            "❌ No data detected. (Is Gazebo/Bridge running? Are you on the right Domain ID?)"
        )
    else:
        print("✅ Discovered active topics:")
        for key in sorted(discovered_keys):
            # Clean up the output to make it look like standard ROS 2 topics
            if f"{domain_id}/rt" in key:
                ros_topic = key.split(f"{domain_id}/rt")[-1]
                print(f"  -> {ros_topic}  (ROS 2 Topic)")
            else:
                print(f"  -> {key}  (Other Zenoh Key)")
    print("-" * 50 + "\n")


def on_image_received(msg):
    """Callback function for raw, uncompressed Image messages"""
    global last_msg_time, is_receiving

    last_msg_time = time.time()
    if not is_receiving:
        print(f"[SUCCESS] Actively receiving data on: {TOPIC}")
        is_receiving = True

    try:
        height = msg.height
        width = msg.width
        encoding = msg.encoding

        image_data = msg.data
        if hasattr(image_data, "tobytes"):
            image_bytes = image_data.tobytes()
        elif isinstance(image_data, list):
            image_bytes = bytes(image_data)
        else:
            image_bytes = bytes(image_data)

        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)

        if encoding == "rgb8":
            frame = np_arr.reshape((height, width, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif encoding == "bgr8":
            frame = np_arr.reshape((height, width, 3))
        elif encoding == "bgra8":
            frame = np_arr.reshape((height, width, 4))
        elif encoding == "rgba8":
            frame = np_arr.reshape((height, width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        elif encoding == "mono8":
            frame = np_arr.reshape((height, width))
        else:
            print(f"Warning: Unsupported encoding '{encoding}'. Cannot reshape.")
            return

        if frame is not None:
            cv2.imshow("ZED Camera Stream", frame)
            cv2.waitKey(1)

    except Exception as e:
        print(f"Error processing frame: {e}")


def main():
    # 1. RUN THE SCANNER FIRST
    scan_active_topics(ROUTER_IP, ROUTER_PORT, DOMAIN_ID, scan_time=2.0)

    print(f"Connecting to Zenoh at {ROUTER_IP}:{ROUTER_PORT}...")

    # 2. Initialize Session
    ZenohSession.get_instance(router_ip=ROUTER_IP, router_port=ROUTER_PORT)

    print(f"Subscribing to: {TOPIC} (Domain: {DOMAIN_ID})")

    # 3. Create Subscriber
    sub = ROS2Subscriber(
        topic=TOPIC,
        msg_type=MSG_TYPE,
        domain_id=DOMAIN_ID,
        callback=on_image_received,
        router_ip=ROUTER_IP,
        router_port=ROUTER_PORT,
    )

    last_print_time = time.time()
    global is_receiving

    try:
        print("Stream started. Press Ctrl+C to exit.")
        while True:
            time.sleep(0.5)
            current_time = time.time()

            if not is_receiving:
                if current_time - last_print_time > TIMEOUT_SECONDS:
                    print(f"[WAITING] Looking for topic '{TOPIC}'. No data yet...")
                    last_print_time = current_time
            else:
                if current_time - last_msg_time > TIMEOUT_SECONDS:
                    print(
                        f"\n[WARNING] Data stream interrupted! No messages for {TIMEOUT_SECONDS} seconds."
                    )
                    is_receiving = False
                    cv2.destroyAllWindows()
                    last_print_time = current_time

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sub.close()
        cv2.destroyAllWindows()
        ZenohSession.get_instance().close()


if __name__ == "__main__":
    main()
