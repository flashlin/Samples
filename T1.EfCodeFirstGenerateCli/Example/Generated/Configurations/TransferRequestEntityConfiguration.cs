using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class TransferRequestEntityConfiguration : IEntityTypeConfiguration<TransferRequestEntity>
    {
        public void Configure(EntityTypeBuilder<TransferRequestEntity> builder)
        {
            builder.ToTable("TransferRequest");


            builder.Property(x => x.requestid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.requestrefno)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.requestdate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.amount)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.istakeremaining)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.fromproduct)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.toproduct)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.requesterid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.requestername)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.status)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.remark)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Mode)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Wonglaitype)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(3)
            ;

        }
    }
}
