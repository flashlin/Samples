using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SuspendedByCompanyLogEntityConfiguration : IEntityTypeConfiguration<SuspendedByCompanyLogEntity>
    {
        public void Configure(EntityTypeBuilder<SuspendedByCompanyLogEntity> builder)
        {
            builder.ToTable("SuspendedByCompanyLog");

            builder.HasKey(x => x.SuspendedByCompanyLogId);

            builder.Property(x => x.SuspendedByCompanyLogId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.PreviousSuspended)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.PreviousSuspendedByCompany)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.SuspendedBy)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SuspendedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
