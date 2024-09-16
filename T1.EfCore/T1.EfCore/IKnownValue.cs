namespace T1.EfCore;

public interface IKnownValue
{
    IEnumerable<ConstantValue> GetConstantValues();
    IEnumerable<PropertyValue> GetPropertyValues();
}