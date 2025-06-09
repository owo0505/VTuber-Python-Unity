// Assets/Scripts/HeadPoseDriver.cs
using UnityEngine;

public class HeadPoseDriver : MonoBehaviour
{
    [Tooltip("拖入 Avatar 裡的 Head Bone (Transform)")]
    public Transform headBone;

    /// <summary>
    /// value: roll/pitch/yaw (deg)
    /// </summary>
    public void SetHeadPose(float roll, float pitch, float yaw)
    {
        // Unity Euler 顺序是 ZXY，依次传 z=roll,x=pitch,y=yaw
        headBone.localRotation = Quaternion.Euler(pitch, yaw, roll);
    }
}
