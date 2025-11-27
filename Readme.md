# Como correr el Server

Corre el archivo server.py

Nota: esta parte del código debe estar en agente py

```
def run_simulation_return_history(N:int, T:int, capacity:int, p_good:float=0.5, seed:int=42, max_ticks:int=2000) -> Dict[str, Any]:
    pars = {'N': N, 'T': T, 'capacity': capacity, 'p_good': p_good, 'seed': seed, 'max_ticks': max_ticks}
    model = CampoModel(pars)
    model.run()
    return model.history
```

Corre en modo local en el puerto http://127.0.0.1:5000/simulated

Crea el cliente en Unity C#

```
using System;
using System.Collections;
using System.IO;
using System.Net;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class DataJson : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created

    public class SimRequest
    {
        public int N = 12;
        public int T = 4;
        public int capacity = 10;
        public float p_good = 0.5f;
        public int seed = 42;
        public int max_ticks = 500;
    }
    void Start()
    {
        StartCoroutine(postRequest("http://127.0.0.1:5000/simulated"));
    }
    IEnumerator postRequest(string url)
    {
        SimRequest payload = new SimRequest {
            N = 8,
            T = 4,
            capacity = 6,
            p_good = 0.5f,
            seed = 42,
            max_ticks = 20
        };

        string json = JsonUtility.ToJson(payload);

        var uwr = new UnityWebRequest(url,"POST");
        byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(json);
        uwr.uploadHandler = (UploadHandler)new UploadHandlerRaw(jsonToSend);
        uwr.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
        uwr.SetRequestHeader("Content-Type", "application/json");

        yield return uwr.SendWebRequest();

        if (uwr.result == UnityWebRequest.Result.ConnectionError)
        {
            Debug.Log("Error While Sending: " + uwr.error);
        }
        else
        {
            Debug.Log("Received: " + uwr.downloadHandler.text);
        }
    }

}
```