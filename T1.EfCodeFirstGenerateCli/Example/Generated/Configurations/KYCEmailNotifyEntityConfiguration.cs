using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class KYCEmailNotifyEntityConfiguration : IEntityTypeConfiguration<KYCEmailNotifyEntity>
    {
        public void Configure(EntityTypeBuilder<KYCEmailNotifyEntity> builder)
        {
            builder.ToTable("KYCEmailNotify");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.email)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.KYCExpiryDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.status)
                .HasColumnType("bit")
            ;

        }
    }
}
