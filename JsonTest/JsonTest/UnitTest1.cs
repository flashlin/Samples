using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using ExpectedObjects;
using FluentAssertions;
using NSubstitute;
using T1.Standard.Threads;

namespace JsonTest
{
    public class Tests
    {
        public class User
        {
            public int Id { get; set; }
            public string Name { get; set; }
            public DeviceType Device { get; set; }
        }

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test1()
        {
            var userJson = "{ \"Id\": 123, \"Name\": 123 }";

            var options = new JsonSerializerOptions
            {
                Converters =
                {
                    new StringConverter(),
                    //new DictionaryTKeyEnumTValueConverter(),
                }
            };
            var user = JsonSerializer.Deserialize<User>(userJson, options)!;
            Assert.AreEqual(123, user.Id);
        }

        [Test]
        public void TestTwoListWhenOrderSame()
        {
            var expected = new[]
            {
                new User()
                {
                    Device = DeviceType.Mobile,
                },
                new User()
                {
                    Device = DeviceType.Desktop,
                }
            };

            var actual = new[]
            {
                new User()
                {
                    Device = DeviceType.Mobile,
                },
                new User()
                {
                    Device = DeviceType.Desktop,
                },
            };

            CollectionAsserter.BeEqualMatch(expected, actual);
        }
        
        [Test]
        public void TestTwoListWhenOrderDifferent()
        {
            var expected = new[]
            {
                new User()
                {
                    Device = DeviceType.Desktop,
                },
                new User()
                {
                    Device = DeviceType.Mobile,
                },
            };

            var actual = new[]
            {
                new User()
                {
                    Device = DeviceType.Mobile,
                },
                new User()
                {
                    Device = DeviceType.Desktop,
                },
            };

            CollectionAsserter.BeEqualMatch(expected, actual);
        }
        
        
        
        [Test]
        public void TestAsyncHelper()
        {
            var expected = 111;

            var myRepo = Substitute.For<IMyRepo>();
            myRepo.TestAsync(Arg.Any<User>()).Returns(Task.FromResult(111));
            
            
            var service = new MyService()
            {
                MyRepo = myRepo
            };
            
            

            var actual = service.Test(new User()
            {
                Id = 1
            });
            
            expected.Should()
                .Be(actual);
        }
        
        


        public class MyService
        {
            public IMyRepo MyRepo { get; set; } 
            
            public int Test(User user)
            {
                var rc = AsyncHelper.RunSync(() => MyRepo.TestAsync(user));
                return rc;
            }
        }

        public interface IMyRepo
        {
            Task<int> TestAsync(User user);
        }

        public class MyRepo : IMyRepo
        {
            public async Task<int> TestAsync(User user)
            {
                return await Task.FromResult(123);
            }
        }
    }

    public static class CollectionAsserter
    {
        public static void BeEqualMatch<T>(IEnumerable<T> expectedList, IEnumerable<T> actualList)
        {
            var expectedArr = expectedList.ToArray();
            var actualArr = actualList.ToArray();
            expectedArr.Length.Should().Be(actualArr.Length);
            var index = 0;
            foreach (var (actual, expected) in actualArr.Zip(expectedArr))
            {
                actual.Should().BeEquivalentTo(expected, $"actual[{index}]");
                index++;
            }
        }
    }

    public enum DeviceType
    {
        Mobile,
        Desktop
    }

    public class StringConverter : JsonConverter<string>
    {
        public override string? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            try
            {
                return reader.GetString();
            }
            catch
            {
                return string.Empty;
            }
        }

        public override void Write(Utf8JsonWriter writer, string value, JsonSerializerOptions options)
        {
            throw new NotImplementedException();
        }
    }


    public class DictionaryTKeyEnumTValueConverter : JsonConverterFactory
    {
        public override bool CanConvert(Type typeToConvert)
        {
            if (!typeToConvert.IsGenericType)
            {
                return false;
            }

            if (typeToConvert.GetGenericTypeDefinition() != typeof(Dictionary<,>))
            {
                return false;
            }

            return typeToConvert.GetGenericArguments()[0].IsEnum;
        }

        public override JsonConverter CreateConverter(
            Type type,
            JsonSerializerOptions options)
        {
            Type keyType = type.GetGenericArguments()[0];
            Type valueType = type.GetGenericArguments()[1];

            JsonConverter converter = (JsonConverter)Activator.CreateInstance(
                typeof(DictionaryEnumConverterInner<,>).MakeGenericType(
                    new Type[] { keyType, valueType }),
                BindingFlags.Instance | BindingFlags.Public,
                binder: null,
                args: new object[] { options },
                culture: null)!;

            return converter;
        }

        private class DictionaryEnumConverterInner<TKey, TValue> : JsonConverter<Dictionary<TKey, TValue>>
            where TKey : struct, Enum
        {
            private readonly JsonConverter<TValue> _valueConverter;
            private readonly Type _keyType;
            private readonly Type _valueType;

            public DictionaryEnumConverterInner(JsonSerializerOptions options)
            {
                // For performance, use the existing converter if available.
                _valueConverter = (JsonConverter<TValue>)options
                    .GetConverter(typeof(TValue));

                // Cache the key and value types.
                _keyType = typeof(TKey);
                _valueType = typeof(TValue);
            }

            public override Dictionary<TKey, TValue> Read(
                ref Utf8JsonReader reader,
                Type typeToConvert,
                JsonSerializerOptions options)
            {
                if (reader.TokenType != JsonTokenType.StartObject)
                {
                    throw new JsonException();
                }

                var dictionary = new Dictionary<TKey, TValue>();

                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                    {
                        return dictionary;
                    }

                    // Get the key.
                    if (reader.TokenType != JsonTokenType.PropertyName)
                    {
                        throw new JsonException();
                    }

                    string? propertyName = reader.GetString();

                    // For performance, parse with ignoreCase:false first.
                    if (!Enum.TryParse(propertyName, ignoreCase: false, out TKey key) &&
                        !Enum.TryParse(propertyName, ignoreCase: true, out key))
                    {
                        throw new JsonException(
                            $"Unable to convert \"{propertyName}\" to Enum \"{_keyType}\".");
                    }

                    // Get the value.
                    TValue value;
                    if (_valueConverter != null)
                    {
                        reader.Read();
                        value = _valueConverter.Read(ref reader, _valueType, options)!;
                    }
                    else
                    {
                        value = JsonSerializer.Deserialize<TValue>(ref reader, options)!;
                    }

                    // Add to dictionary.
                    dictionary.Add(key, value);
                }

                throw new JsonException();
            }

            public override void Write(
                Utf8JsonWriter writer,
                Dictionary<TKey, TValue> dictionary,
                JsonSerializerOptions options)
            {
                writer.WriteStartObject();

                foreach ((TKey key, TValue value) in dictionary)
                {
                    var propertyName = key.ToString();
                    writer.WritePropertyName
                        (options.PropertyNamingPolicy?.ConvertName(propertyName) ?? propertyName);

                    if (_valueConverter != null)
                    {
                        _valueConverter.Write(writer, value, options);
                    }
                    else
                    {
                        JsonSerializer.Serialize(writer, value, options);
                    }
                }

                writer.WriteEndObject();
            }
        }
    }
}