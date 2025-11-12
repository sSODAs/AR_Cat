using System;
using System.Collections.Concurrent;
using UnityEngine;

public class UnityMainThread : MonoBehaviour
{
    private static UnityMainThread _instance;
    private static readonly ConcurrentQueue<Action> _actions = new ConcurrentQueue<Action>();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Initialize()
    {
        if (_instance == null)
        {
            _instance = new GameObject("UnityMainThread").AddComponent<UnityMainThread>();
            DontDestroyOnLoad(_instance.gameObject);
        }
    }

    private void Update()
    {
        while (_actions.TryDequeue(out var action))
        {
            action.Invoke();
        }
    }

    public static void Enqueue(Action action)
    {
        _actions.Enqueue(action);
    }
}