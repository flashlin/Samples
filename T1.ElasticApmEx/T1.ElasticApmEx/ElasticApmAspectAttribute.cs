using System.Diagnostics;
using System.Text.Json;
using AspectInjector.Broker;
using Elastic.Apm;
using Elastic.Apm.Api;
using Target = AspectInjector.Broker.Target;

namespace T1.ElasticApmEx;

[Aspect(Scope.PerInstance)]
[Injection(typeof(ElasticApmAspectAttribute))]
public class ElasticApmAspectAttribute : Attribute
{
    private readonly string _type;

    public ElasticApmAspectAttribute() 
        : this("code")
    {
    }

    public ElasticApmAspectAttribute(string type)
    {
        _type = type;
    }

    [Advice(Kind.Around, Targets = Target.Method)]
    public object Handle(
        [Argument(Source.Target)] Func<object[], object> method,
        [Argument(Source.Arguments)] object[] args,
        [Argument(Source.Name)] string methodName,
        [Argument(Source.Type)] Type declaringType)
    {
        var tracer = Agent.Tracer;
        var currentTransaction = tracer.CurrentTransaction;
        var fullMethodName = $"{declaringType.FullName}.{methodName}";

        if (currentTransaction == null)
        {
            // 外層 -> Transaction
            var transaction = tracer.StartTransaction(fullMethodName, _type);
            return InvokeWithinTransaction(method, args, transaction, fullMethodName);
        }
        else
        {
            // 內層 -> Span
            var currentSpan = tracer.CurrentSpan;
            var span = ((IExecutionSegment)currentSpan ?? currentTransaction).StartSpan(fullMethodName, _type);
            return InvokeWithinSpan(method, args, span, fullMethodName);
        }
    }

    private object InvokeWithinTransaction(Func<object[], object> method, object[] args, ITransaction transaction,
        string name)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            if (args.Any())
            {
                transaction.SetLabel("args", JsonSerializer.Serialize(args));
            }

            var result = method(args);
            if (result is Task task)
            {
                dynamic dynamicTask = task;
                return WrapAsyncTask(dynamicTask, transaction, stopwatch);
            }

            stopwatch.Stop();
            transaction.SetLabel("duration_ms", stopwatch.ElapsedMilliseconds.ToString());
            return result;
        }
        catch (Exception ex)
        {
            transaction.CaptureException(ex);
            throw;
        }
        finally
        {
            transaction.End();
        }
    }

    private object InvokeWithinSpan(Func<object[], object> method, object[] args, ISpan span, string name)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            if (args.Any())
            {
                span.SetLabel("args", JsonSerializer.Serialize(args));
            }

            var result = method(args);
            if (result is Task task)
            {
                dynamic dynamicTask = task;
                return WrapAsyncTask(dynamicTask, span, stopwatch);
            }
            stopwatch.Stop();
            span.SetLabel("duration_ms", stopwatch.ElapsedMilliseconds.ToString());
            return result;
        }
        catch (Exception ex)
        {
            span.CaptureException(ex);
            throw;
        }
        finally
        {
            span.End();
        }
    }

    private async Task WrapAsyncTask(Task task, ITransaction transaction, Stopwatch sw)
    {
        try
        {
            await task.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            transaction.CaptureException(ex);
            throw;
        }
        finally
        {
            sw.Stop();
            transaction.SetLabel("duration_ms", sw.ElapsedMilliseconds.ToString());
            transaction.End();
        }
    }

    private async Task WrapAsyncTask(Task task, ISpan span, Stopwatch sw)
    {
        try
        {
            await task.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            span.CaptureException(ex);
            throw;
        }
        finally
        {
            sw.Stop();
            span.SetLabel("duration_ms", sw.ElapsedMilliseconds.ToString());
            span.End();
        }
    }

    private async Task<T> WrapAsyncTask<T>(Task<T> task, ITransaction transaction, Stopwatch sw)
    {
        try
        {
            var result = await task.ConfigureAwait(false);
            return result;
        }
        catch (Exception ex)
        {
            transaction.CaptureException(ex);
            throw;
        }
        finally
        {
            sw.Stop();
            transaction.SetLabel("duration_ms", sw.ElapsedMilliseconds.ToString());
            transaction.End();
        }
    }

    private async Task<T> WrapAsyncTask<T>(Task<T> task, ISpan span, Stopwatch sw)
    {
        try
        {
            var result = await task.ConfigureAwait(false);
            return result;
        }
        catch (Exception ex)
        {
            span.CaptureException(ex);
            throw;
        }
        finally
        {
            sw.Stop();
            span.SetLabel("duration_ms", sw.ElapsedMilliseconds.ToString());
            span.End();
        }
    }
}