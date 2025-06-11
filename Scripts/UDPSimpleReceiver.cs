// Assets/Scripts/UDPSimpleReceiver.cs
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

[RequireComponent(typeof(BlendShapeDriver))]
public class UDPSimpleReceiver : MonoBehaviour
{
    public int port = 9000;

    UdpClient udp;
    Thread thread;
    Queue<string> queue = new Queue<string>();
    object queueLock = new object();

    BlendShapeDriver blendDriver;
    HeadPoseDriver headDriver;

    void Awake()
    {
        blendDriver = GetComponent<BlendShapeDriver>();
        headDriver = GetComponent<HeadPoseDriver>();
        if (headDriver == null)
            Debug.LogWarning("HeadPoseDriver！ Not Found");
    }

    void Start()
    {
        udp = new UdpClient(port);
        thread = new Thread(ReceiveLoop) { IsBackground = true };
        thread.Start();
    }

    void ReceiveLoop()
    {
        var ep = new IPEndPoint(IPAddress.Any, port);
        while (true)
        {
            try
            {
                var data = udp.Receive(ref ep);
                var msg = Encoding.UTF8.GetString(data).Trim();
                lock (queueLock)
                    queue.Enqueue(msg);
            }
            catch { }
        }
    }

    void Update()
    {
        lock (queueLock)
        {
            while (queue.Count > 0)
            {
                var msg = queue.Dequeue();
                var parts = msg.Split(' ');
                if (parts.Length == 7
                    && float.TryParse(parts[0], out float blinkL)
                    && float.TryParse(parts[1], out float blinkR)
                    && float.TryParse(parts[2], out float mar)
                    && float.TryParse(parts[3], out float dist)
                    && float.TryParse(parts[4], out float roll)
                    && float.TryParse(parts[5], out float pitch)
                    && float.TryParse(parts[6], out float yaw))
                {
                    // clamp & apply
                    blendDriver.SetEyeBlinks(
                      Mathf.Clamp(blinkL, 0, 100f),
                      Mathf.Clamp(blinkR, 0, 100f)
                    );
                    blendDriver.SetMouth(
                      Mathf.Clamp(mar, 0, 100f),
                      Mathf.Clamp(dist, 0, 100f)
                    );
                    if (headDriver != null)
                        headDriver.SetHeadPose(roll, pitch, yaw);
                }

                else if (parts.Length == 1)
                {
                    string emo = parts[0].ToLower();
                    if (emo == "angry" || emo == "sad" ||
                        emo == "happy" || emo == "neutral")
                    {
                        blendDriver.SetEmotion(emo);
                    }
                }
            }
        }
    }

    void OnDestroy()
    {
        thread?.Abort();
        udp?.Close();
    }
}
