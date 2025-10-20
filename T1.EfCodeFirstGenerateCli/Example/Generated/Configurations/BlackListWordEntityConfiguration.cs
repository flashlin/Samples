using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BlackListWordEntityConfiguration : IEntityTypeConfiguration<BlackListWordEntity>
    {
        public void Configure(EntityTypeBuilder<BlackListWordEntity> builder)
        {
            builder.ToTable("BlackListWord");

            builder.HasKey(x => x.BlackListWordID);

            builder.Property(x => x.BlackListWordID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.BlackListWord)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.BlackListWordType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Status)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.LastModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
