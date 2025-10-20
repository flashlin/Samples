using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SelfExclusionEmailNotifyEntityConfiguration : IEntityTypeConfiguration<SelfExclusionEmailNotifyEntity>
    {
        public void Configure(EntityTypeBuilder<SelfExclusionEmailNotifyEntity> builder)
        {
            builder.ToTable("SelfExclusionEmailNotify");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.email)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SelfExclusionExpiredDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.status)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.firstname)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.period)
                .HasColumnType("int(10,0)")
            ;

        }
    }
}
