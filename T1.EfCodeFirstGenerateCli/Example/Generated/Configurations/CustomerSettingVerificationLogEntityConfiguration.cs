using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerSettingVerificationLogEntityConfiguration : IEntityTypeConfiguration<CustomerSettingVerificationLogEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerSettingVerificationLogEntity> builder)
        {
            builder.ToTable("CustomerSettingVerificationLog");


            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Username)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.RoleId)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
            ;

            builder.Property(x => x.ProductType)
                .HasColumnType("smallint(5,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TableName)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CustomerCreatedDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.HasTransaction)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
