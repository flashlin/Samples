using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class transEntityConfiguration : IEntityTypeConfiguration<transEntity>
    {
        public void Configure(EntityTypeBuilder<transEntity> builder)
        {
            builder.ToTable("trans");

            builder.HasKey(x => x.rid);

            builder.Property(x => x.transcode)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.rid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

        }
    }
}
