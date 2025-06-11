// Assets/Scripts/BlendShapeDriver.cs
using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(SkinnedMeshRenderer))]
public class BlendShapeDriver : MonoBehaviour
{
    SkinnedMeshRenderer smr;
    int idxLeft, idxRight, idxJo, idxSo;

    Dictionary<string, int[]> emotionMap;
    string currentEmotion = "neutral";

    void Awake()
    {
        smr = GetComponent<SkinnedMeshRenderer>();

        idxLeft = smr.sharedMesh.GetBlendShapeIndex("PS_eyeBlinkLeft");
        idxRight = smr.sharedMesh.GetBlendShapeIndex("PS_eyeBlinkRight");
        idxJo = smr.sharedMesh.GetBlendShapeIndex("PS_jawOpen");
        idxSo = smr.sharedMesh.GetBlendShapeIndex("Mouth_Smile_Open");

        emotionMap = new Dictionary<string, int[]>() {
            { "angry", new [] {
                smr.sharedMesh.GetBlendShapeIndex("Eyebrow_Angry"),
                smr.sharedMesh.GetBlendShapeIndex("Eye_Half_Angry"),
                smr.sharedMesh.GetBlendShapeIndex("Eye_Other_Deadeye")
            }},
            { "sad", new [] {
                smr.sharedMesh.GetBlendShapeIndex("Eyebrow_Sad"),
                smr.sharedMesh.GetBlendShapeIndex("Eye_Half_Sad"),
                smr.sharedMesh.GetBlendShapeIndex("Other_Tears_Many")
            }},
            { "happy", new [] {
                smr.sharedMesh.GetBlendShapeIndex("Mouth_Cat"),
                smr.sharedMesh.GetBlendShapeIndex("Eye_Surprise_Small"),
                smr.sharedMesh.GetBlendShapeIndex("Eye_Other_Star")
            }},
            { "neutral", new int[0] }
        };
    }

    /// <summary>
    /// Blink weight（range 0~100）
    /// </summary>
    public void SetEyeBlinks(float leftValue, float rightValue)
    {
        smr.SetBlendShapeWeight(idxLeft, leftValue);
        smr.SetBlendShapeWeight(idxRight, rightValue);
    }

    public void SetMouth(float marvalue, float distanceValue)
    {
        smr.SetBlendShapeWeight(idxJo, marvalue);
        smr.SetBlendShapeWeight(idxSo, distanceValue);
    }

    public void SetEmotion(string emo)
    {
        if (emo == currentEmotion) return;

        // old emotion 
        foreach (var idx in emotionMap[currentEmotion])
            if (idx >= 0) smr.SetBlendShapeWeight(idx, 0f);

        // new emotion 
        foreach (var idx in emotionMap[emo])
            if (idx >= 0) smr.SetBlendShapeWeight(idx, 100f);

        currentEmotion = emo;
    }
}
