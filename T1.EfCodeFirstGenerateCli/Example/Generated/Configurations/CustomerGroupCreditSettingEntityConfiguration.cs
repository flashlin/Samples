using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerGroupCreditSettingEntityConfiguration : IEntityTypeConfiguration<CustomerGroupCreditSettingEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerGroupCreditSettingEntity> builder)
        {
            builder.ToTable("CustomerGroupCreditSetting");

            builder.HasKey(x => x.CustomerGroupId);

            builder.Property(x => x.CustomerGroupId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.GivenCredit)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.SmaCreditLimit)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.MaxCredit)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
