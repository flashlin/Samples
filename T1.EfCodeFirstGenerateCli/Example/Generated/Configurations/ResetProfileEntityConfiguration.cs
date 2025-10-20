using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class ResetProfileEntityConfiguration : IEntityTypeConfiguration<ResetProfileEntity>
    {
        public void Configure(EntityTypeBuilder<ResetProfileEntity> builder)
        {
            builder.ToTable("ResetProfile");

            builder.HasKey(x => x.ProfileId);

            builder.Property(x => x.ProfileId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Mon)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Tue)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Wed)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Thu)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Fri)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Sat)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Sun)
                .HasColumnType("bit")
                .IsRequired()
            ;

        }
    }
}
