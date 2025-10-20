using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SB_RobotBetIPEntityConfiguration : IEntityTypeConfiguration<SB_RobotBetIPEntity>
    {
        public void Configure(EntityTypeBuilder<SB_RobotBetIPEntity> builder)
        {
            builder.ToTable("SB_RobotBetIP");


            builder.Property(x => x.ID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.IP)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.IsNewIP)
                .HasColumnType("bit")
                .IsRequired()
            ;

        }
    }
}
