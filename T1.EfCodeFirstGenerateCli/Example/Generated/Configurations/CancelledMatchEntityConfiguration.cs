using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CancelledMatchEntityConfiguration : IEntityTypeConfiguration<CancelledMatchEntity>
    {
        public void Configure(EntityTypeBuilder<CancelledMatchEntity> builder)
        {
            builder.ToTable("CancelledMatch");

            builder.HasKey(x => x.MatchResultId);

            builder.Property(x => x.MatchResultId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CancelledDate)
                .HasColumnType("smalldatetime")
                .IsRequired()
            ;

        }
    }
}
