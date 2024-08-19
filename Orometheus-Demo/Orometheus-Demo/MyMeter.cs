using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace Orometheus_Demo;

public class MyMeter
{
    private Counter<int> _requestsCounter;
    private Histogram<float> _requestsHistogram;

    public MyMeter()
    {
        var meter = new Meter("MyApplication");
        _requestsCounter = meter.CreateCounter<int>("Requests");
        _requestsHistogram = meter.CreateHistogram<float>("RequestDuration", unit: "ms");
        // meter.CreateObservableGauge("ThreadCount", () => new[]
        // {
        //     new Measurement<int>(ThreadPool.ThreadCount)
        // });
    }
    
    public void IncrementCounter(string userName)
    {
        _requestsCounter.Add(1, KeyValuePair.Create<string, object?>("name", userName));
    }

    public void Record(Stopwatch stopwatch)
    {
        //var stopwatch = Stopwatch.StartNew();
        _requestsHistogram.Record(stopwatch.ElapsedMilliseconds,
            tag: KeyValuePair.Create<string, object?>("Host", "www.meziantou.net"));
    }
}