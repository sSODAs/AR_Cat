using System.IO;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using System.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.Collections;
using UnityEngine.Networking;
using Mediapipe.Unity;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.HandLandmarker;
using Mediapipe.Tasks.Vision.Core;
using System;
using Mediapipe;

public class HandARController : MonoBehaviour
{
    [Header("AR Foundation")]
    public ARCameraManager cameraManager;
    public ARTrackedImageManager trackedImageManager;

    private bool isHandDetecting = false;

    [Header("Target (Cat Model)")]
    private GameObject activeCatModel;
    private Animator activeCatAnimator;

    [Header("AR Object (AR Toy/Particle System)")]
    public GameObject arCatToy;

    [Header("Debug UI")]
    public TextMeshProUGUI debugText;

    [Header("MediaPipe Model")]
    public string modelName = "hand_landmarker.task";

    [Header("Hand Skeleton Debug")]
    public bool drawHandLines = false;
    public Material lineMaterial;
    public float lineWidth = 0.005f;
    public float depthScale = 0.5f;

    [Header("Interaction Settings")]
    public bool enableInteraction = true;
    public float detectInterval = 0.08f;
    public bool isCameraMirrored = true;
    public float fingerBendThreshold = 0.1f;
    public float pokeDistanceThreshold = 0.3f;

    [Header("Tuning")]
    public float rotationSpeed = 10.0f;
    public float minDistance = 0.25f;
    public float maxDistance = 3.0f;
    public float walkSpeed = 1.5f;

    private float lastDetectTime = 0f;
    private HandLandmarker handLandmarker;

    private Camera arCamera;
    private LineRenderer handLineRenderer;
    private string actualModelPath;

    private GameObject catAnchorObject;
    private bool isCatAnchored = false;
    private string currentGesture = "Idle";

    // Gesture smoothing
    private Queue<string> gestureBuffer = new Queue<string>();
    private int bufferSize = 5;

    // --- Animation tracking ---
    private float gestureHoldTime = 0f;
    private string lastGesture = "NoHand";
    private float idleSwitchTime = 0f;
    private float idleInterval = 120f; // 120 วินาที = 2 นาที
    private string currentAnim = "Idle";
    private string[] idleAnimations = { "Idle", "Idle_a", "Idle_b", "Idle_c", "Sleep" };
    private string[] lightIdleAnimations = { "Idle", "Idle_a", "Idle_b", "Idle_c" };

    private void OnDisable()
    {
        if (cameraManager != null) cameraManager.frameReceived -= OnCameraFrameReceived;
        if (trackedImageManager != null) trackedImageManager.trackedImagesChanged -= OnTrackedImagesChanged;

        if (handLandmarker != null) { try { handLandmarker.Close(); } catch { } handLandmarker = null; }
        if (handLineRenderer != null && handLineRenderer.gameObject != null) { Destroy(handLineRenderer.gameObject); handLineRenderer = null; }
        if (catAnchorObject != null) { Destroy(catAnchorObject); catAnchorObject = null; }
    }

    IEnumerator Start()
    {
        debugText?.SetText("Initializing...");

        if (cameraManager == null || trackedImageManager == null) { debugText?.SetText("Error: AR Managers not assigned"); yield break; }

        arCamera = cameraManager.GetComponentInChildren<Camera>();
        if (arCamera == null) { debugText?.SetText("Error: AR Camera not found!"); yield break; }
        if (arCatToy != null) arCatToy.SetActive(false);

        if (drawHandLines)
        {
            GameObject lineObj = new GameObject("HandDebugLine");
            lineObj.transform.SetParent(transform);
            handLineRenderer = lineObj.AddComponent<LineRenderer>();
            handLineRenderer.material = lineMaterial ?? new Material(Shader.Find("Sprites/Default"));
            handLineRenderer.startWidth = lineWidth;
            handLineRenderer.endWidth = lineWidth;
            handLineRenderer.useWorldSpace = true;
            handLineRenderer.enabled = false;
        }

        yield return new WaitUntil(() => cameraManager.permissionGranted);

        if (catAnchorObject != null) Destroy(catAnchorObject);
        catAnchorObject = new GameObject("Cat_Anchor_Root");

        debugText?.SetText("Loading AI Model...");
        string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, modelName);
        actualModelPath = Path.Combine(Application.persistentDataPath, modelName);
        if (!File.Exists(actualModelPath))
        {
            using (var www = UnityWebRequest.Get(streamingAssetsPath))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success) { debugText?.SetText($"AI MODEL LOAD FAILED: {www.error}"); yield break; }
                try { File.WriteAllBytes(actualModelPath, www.downloadHandler.data); } catch (Exception e) { debugText?.SetText($"Write model failed: {e.Message}"); yield break; }
            }
        }
        if (!File.Exists(actualModelPath)) { debugText?.SetText("AI FILE NOT FOUND"); yield break; }

        var baseOptions = new BaseOptions(modelAssetPath: actualModelPath);
        var options = new HandLandmarkerOptions(
            baseOptions: baseOptions,
            runningMode: Mediapipe.Tasks.Vision.Core.RunningMode.LIVE_STREAM,
            numHands: 1,
            minHandDetectionConfidence: 0.7f,
            minHandPresenceConfidence: 0.7f,
            minTrackingConfidence: 0.5f,
            resultCallback: OnHandLandmarkerResult
        );

        try { handLandmarker = HandLandmarker.CreateFromOptions(options); debugText?.SetText("AI LOADED"); }
        catch (Exception e) { debugText?.SetText($"AI FAILED: {e.Message}"); yield break; }

        trackedImageManager.trackedImagesChanged += OnTrackedImagesChanged;
        cameraManager.frameReceived += OnCameraFrameReceived;
    }

    private void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        if (Time.time - lastDetectTime < detectInterval) return;
        lastDetectTime = Time.time;

        if (isHandDetecting) return;
        if (handLandmarker == null) { debugText?.SetText("NO HAND MODEL"); return; }
        if (!cameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage)) { debugText?.SetText("NO CPU IMAGE"); return; }

        isHandDetecting = true;

        var conversionParams = new XRCpuImage.ConversionParams(cpuImage, TextureFormat.RGBA32, XRCpuImage.Transformation.None);
        int bufferSize = cpuImage.GetConvertedDataSize(conversionParams);
        var buffer = new NativeArray<byte>(bufferSize, Allocator.Temp);

        try
        {
            cpuImage.Convert(conversionParams, buffer);
            var image = new Mediapipe.Image(ImageFormat.Types.Format.Srgba, conversionParams.outputDimensions.x, conversionParams.outputDimensions.y, conversionParams.outputDimensions.x * 4, buffer);
            long ts = (long)(cpuImage.timestamp * 1000.0);
            handLandmarker.DetectAsync(image, ts);
        }
        catch (Exception e) { debugText?.SetText($"CONVERT FAILED: {e.Message}"); isHandDetecting = false; }
        finally { if (buffer.IsCreated) buffer.Dispose(); cpuImage.Dispose(); }
    }

    private void OnHandLandmarkerResult(HandLandmarkerResult result, Mediapipe.Image image, long timestamp)
    {
        UnityMainThread.Enqueue(() =>
        {
            if (activeCatModel == null)
            {
                debugText?.SetText("Waiting for cat target...");
                isHandDetecting = false;
                return;
            }

            try
            {
                Vector3 fingerWorldPos = Vector3.zero;

                if (result.handLandmarks != null && result.handLandmarks.Count > 0)
                {
                    var firstHand = result.handLandmarks[0];
                    if (firstHand.landmarks != null && firstHand.landmarks.Count >= 21)
                    {
                        string detectedGesture = DetectHandGesture(firstHand);
                        debugText?.SetText($"HAND DETECTED | GESTURE: {detectedGesture}");

                        var fingerTip = firstHand.landmarks[8];
                        float catDistance = Vector3.Distance(arCamera.transform.position, activeCatModel.transform.position);
                        catDistance = Mathf.Clamp(catDistance, minDistance, maxDistance);

                        Vector3 screenPos = new Vector3(
                            (isCameraMirrored ? 1f - fingerTip.x : fingerTip.x) * UnityEngine.Screen.width,
                            (1f - fingerTip.y) * UnityEngine.Screen.height,
                            catDistance + fingerTip.z * depthScale
                        );
                        fingerWorldPos = arCamera.ScreenToWorldPoint(screenPos);

                        UpdateCatRotation(fingerWorldPos);
                        UpdateCatAnimation(detectedGesture, fingerWorldPos);

                        if (drawHandLines && handLineRenderer != null)
                        {
                            handLineRenderer.enabled = true;
                            handLineRenderer.positionCount = firstHand.landmarks.Count;
                            for (int i = 0; i < firstHand.landmarks.Count; i++)
                            {
                                var lm = firstHand.landmarks[i];
                                float zFinal = catDistance + lm.z * depthScale;
                                zFinal = Mathf.Clamp(zFinal, minDistance, maxDistance);

                                Vector3 posScreen = new Vector3(
                                    (isCameraMirrored ? 1f - lm.x : lm.x) * UnityEngine.Screen.width,
                                    (1f - lm.y) * UnityEngine.Screen.height,
                                    zFinal
                                );
                                handLineRenderer.SetPosition(i, arCamera.ScreenToWorldPoint(posScreen));
                            }
                        }
                        else if (handLineRenderer != null)
                        {
                            handLineRenderer.enabled = false;
                        }
                    }
                }

                // ถ้าไม่เจอมือหรือไม่มี gesture ที่ trigger animation → สุ่ม Idle
                if (fingerWorldPos == Vector3.zero)
                {
                    if (Time.time - idleSwitchTime > idleInterval)
                    {
                        string randomIdle = idleAnimations[UnityEngine.Random.Range(0, idleAnimations.Length)];
                        activeCatAnimator.SetTrigger(randomIdle);
                        currentAnim = randomIdle;
                        idleSwitchTime = Time.time;
                    }
                }
            }
            catch (Exception e)
            {
                debugText?.SetText($"Result error: {e.Message}");
            }

            isHandDetecting = false;
        });
    }

    private void AddGestureToBuffer(string gesture)
    {
        gestureBuffer.Enqueue(gesture);
        if (gestureBuffer.Count > bufferSize) gestureBuffer.Dequeue();
    }

    private string GetSmoothedGesture()
    {
        return gestureBuffer
            .GroupBy(g => g)
            .OrderByDescending(g => g.Count())
            .FirstOrDefault()?.Key ?? "Unknown";
    }

    private string DetectHandGesture(Mediapipe.Tasks.Components.Containers.NormalizedLandmarks hand)
    {
        var landmarks = hand.landmarks;
        if (landmarks == null || landmarks.Count < 21) return "Unknown";

        Vector3 GetVector(Mediapipe.Tasks.Components.Containers.NormalizedLandmark lm) => new Vector3(lm.x, lm.y, lm.z);

        var wrist = GetVector(landmarks[0]);
        var thumbTip = GetVector(landmarks[4]);
        var indexBase = GetVector(landmarks[5]);
        var indexTip = GetVector(landmarks[8]);
        var middleBase = GetVector(landmarks[9]);
        var middleTip = GetVector(landmarks[12]);
        var ringBase = GetVector(landmarks[13]);
        var ringTip = GetVector(landmarks[16]);
        var pinkyBase = GetVector(landmarks[17]);
        var pinkyTip = GetVector(landmarks[20]);

        bool IsFingerStraight(Vector3 basePos, Vector3 tipPos)
        {
            Vector3 dir = tipPos - basePos;
            Vector3 wristDir = basePos - wrist;
            float angle = Vector3.Angle(dir, wristDir);
            return angle < 35f;
        }

        bool IsThumbStraight()
        {
            float dist = Vector3.Distance(thumbTip, indexBase);
            return dist > 0.05f;
        }

        bool thumbStraight = IsThumbStraight();
        bool indexStraight = IsFingerStraight(indexBase, indexTip);
        bool middleStraight = IsFingerStraight(middleBase, middleTip);
        bool ringStraight = IsFingerStraight(ringBase, ringTip);
        bool pinkyStraight = IsFingerStraight(pinkyBase, pinkyTip);

        string gesture = "Unknown";
        int straightCount = (indexStraight ? 1 : 0) + (middleStraight ? 1 : 0) + (ringStraight ? 1 : 0) + (pinkyStraight ? 1 : 0);

        if (straightCount == 0 && !thumbStraight) gesture = "Fist";
        else if (straightCount == 4 && thumbStraight) gesture = "OpenPalm";
        else if (indexStraight && !middleStraight && !ringStraight && !pinkyStraight) gesture = "Pointing";
        else if (indexStraight && middleStraight && !ringStraight && !pinkyStraight) gesture = "Peace";
        else gesture = "Unknown";

        AddGestureToBuffer(gesture);
        return GetSmoothedGesture();
    }
    private void UpdateCatAnimation(string gesture, Vector3 fingerWorldPos)
    {
        if (activeCatAnimator == null || arCamera == null) return;

        string catAction = currentAnim; // สำหรับ debug

        if (gesture != "NoHand" && gesture != "Unknown")
        {
            // Update gesture hold time
            if (gesture == lastGesture) gestureHoldTime += Time.deltaTime;
            else gestureHoldTime = 0f;
            lastGesture = gesture;

            switch (gesture)
            {
                case "OpenPalm":
                    if (gestureHoldTime > 0.3f && currentAnim != "Play")
                    {   
                        activeCatAnimator.ResetTrigger("Walk");
                        activeCatAnimator.SetTrigger("Play");
                        currentAnim = "Play";
                        gestureHoldTime = 0f;
                    }
                    break;

                case "Pointing":
                    if (gestureHoldTime >  0.3f && currentAnim != "Sleep")
                    {   
                        activeCatAnimator.ResetTrigger("Walk");
                        activeCatAnimator.SetTrigger("Sleep");
                        currentAnim = "Sleep";
                        gestureHoldTime = 0f;
                    }
                    break;

                case "Peace":
                case "Fist":
                    if (currentAnim != "Walk")
                    {
                        activeCatAnimator.SetTrigger("Walk");
                        currentAnim = "Walk";
                    }
                    break;
            }
        }
        else // ไม่มีมือ → Idle
        {
            if (currentAnim == "Walk") currentAnim = ""; // reset ถ้าเดินอยู่

            if (Time.time - idleSwitchTime > idleInterval)
            {
                string randomIdle = idleAnimations[UnityEngine.Random.Range(0, idleAnimations.Length)];
                activeCatAnimator.SetTrigger(randomIdle);
                currentAnim = randomIdle;
                idleSwitchTime = Time.time;
            }
        }

        // Update debug text
        UpdateDebugText(gesture, currentAnim, fingerWorldPos);
    }


    private void UpdateDebugText(string gesture, string catAction, Vector3 fingerWorldPos)
    {
        if (debugText == null || arCamera == null) return;

        Vector3 screenPos = arCamera.WorldToScreenPoint(fingerWorldPos);

        debugText.text =
            $"👋 HAND: {gesture}\n" +
            $"🐱 CAT: {catAction}\n" +
            $"📌 Finger Pos: X:{screenPos.x:F0} Y:{screenPos.y:F0} Z:{screenPos.z:F2}";
    }

    private void UpdateCatRotation(Vector3 fingerWorldPos)
    {
        if (activeCatModel == null || catAnchorObject == null) return;

        Vector3 worldDirection = fingerWorldPos - activeCatModel.transform.position;
        worldDirection.y = 0;
        if (worldDirection.sqrMagnitude < 0.0001f) return;

        Quaternion targetRotation = Quaternion.LookRotation(worldDirection);
        activeCatModel.transform.rotation = Quaternion.Slerp(activeCatModel.transform.rotation, targetRotation, Time.deltaTime * rotationSpeed);
    }

    IEnumerator HandleImageAdded(ARTrackedImage newImage)
    {
        yield return null;
        yield return new WaitForSeconds(0.2f);

        if (newImage.trackingState != TrackingState.Tracking) yield break;
        if (isCatAnchored) yield break;

        debugText?.SetText("CAT FOUND & ANCHORED!");
        catAnchorObject.transform.position = newImage.transform.position;
        catAnchorObject.transform.rotation = newImage.transform.rotation;

        GameObject pivot = newImage.gameObject; // Cat_Pivot
        activeCatModel = pivot.transform.Find("cu_cat")?.gameObject;
        if (activeCatModel == null)
        {
            debugText?.SetText("ERROR: cu_cat not found under Cat_Pivot!");
            yield break;
        }

        pivot.transform.SetParent(catAnchorObject.transform, worldPositionStays: false);

        activeCatModel.transform.localPosition = Vector3.zero;
        activeCatModel.transform.localRotation = Quaternion.identity;
        activeCatModel.transform.localScale = Vector3.one;

        activeCatAnimator = activeCatModel.GetComponent<Animator>();
        if (activeCatAnimator != null) activeCatAnimator.applyRootMotion = false;

        activeCatModel.SetActive(true);
        isCatAnchored = true;
    }

    private void OnTrackedImagesChanged(ARTrackedImagesChangedEventArgs eventArgs)
    {
        foreach (var newImage in eventArgs.added) { if (!isCatAnchored) StartCoroutine(HandleImageAdded(newImage)); }

        foreach (var removedImage in eventArgs.removed)
        {
            if (isCatAnchored && activeCatModel != null && activeCatModel.name == removedImage.gameObject.name)
            {
                debugText?.SetText("CAT LOST (Destroying Anchor).");
                if (catAnchorObject != null) Destroy(catAnchorObject);
                activeCatModel = null;
                activeCatAnimator = null;
                catAnchorObject = null;
                isCatAnchored = false;
            }
        }
    }
}
